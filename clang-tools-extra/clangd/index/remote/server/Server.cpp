//===--- Server.cpp - gRPC-based Remote Index Server  ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Feature.h"
#include "Index.pb.h"
#include "MonitoringService.grpc.pb.h"
#include "MonitoringService.pb.h"
#include "Service.grpc.pb.h"
#include "Service.pb.h"
#include "index/Index.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/remote/marshalling/Marshalling.h"
#include "support/Context.h"
#include "support/Logger.h"
#include "support/Shutdown.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <chrono>
#include <grpc++/grpc++.h>
#include <grpc++/health_check_service_interface.h>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#if ENABLE_GRPC_REFLECTION
#include <grpc++/ext/proto_server_reflection_plugin.h>
#endif

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace clang {
namespace clangd {
namespace remote {
namespace {

static constexpr char Overview[] = R"(
This is an experimental remote index implementation. The server opens Dex and
awaits gRPC lookup requests from the client.
)";

llvm::cl::opt<std::string> IndexPath(llvm::cl::desc("<INDEX FILE>"),
                                     llvm::cl::Positional, llvm::cl::Required);

llvm::cl::opt<std::string> IndexRoot(llvm::cl::desc("<PROJECT ROOT>"),
                                     llvm::cl::Positional, llvm::cl::Required);

llvm::cl::opt<Logger::Level> LogLevel{
    "log",
    llvm::cl::desc("Verbosity of log messages written to stderr"),
    values(clEnumValN(Logger::Error, "error", "Error messages only"),
           clEnumValN(Logger::Info, "info", "High level execution tracing"),
           clEnumValN(Logger::Debug, "verbose", "Low level details")),
    llvm::cl::init(Logger::Info),
};

llvm::cl::opt<bool> LogPublic{
    "log-public",
    llvm::cl::desc("Avoid logging potentially-sensitive request details"),
    llvm::cl::init(false),
};

llvm::cl::opt<std::string> LogPrefix{
    "log-prefix",
    llvm::cl::desc("A string that'll be prepended to all log statements. "
                   "Useful when running multiple instances on same host."),
};

llvm::cl::opt<std::string> TraceFile(
    "trace-file",
    llvm::cl::desc("Path to the file where tracer logs will be stored"));

llvm::cl::opt<bool> PrettyPrint{
    "pretty",
    llvm::cl::desc("Pretty-print JSON output in the trace"),
    llvm::cl::init(false),
};

llvm::cl::opt<std::string> ServerAddress(
    "server-address", llvm::cl::init("0.0.0.0:50051"),
    llvm::cl::desc("Address of the invoked server. Defaults to 0.0.0.0:50051"));

llvm::cl::opt<size_t> IdleTimeoutSeconds(
    "idle-timeout", llvm::cl::init(8 * 60),
    llvm::cl::desc("Maximum time a channel may stay idle until server closes "
                   "the connection, in seconds. Defaults to 480."));

llvm::cl::opt<size_t> LimitResults(
    "limit-results", llvm::cl::init(10000),
    llvm::cl::desc("Maximum number of results to stream as a response to "
                   "single request. Limit is to keep the server from being "
                   "DOS'd. Defaults to 10000."));

static Key<grpc::ServerContext *> CurrentRequest;

class RemoteIndexServer final : public v1::SymbolIndex::Service {
public:
  RemoteIndexServer(clangd::SymbolIndex &Index, llvm::StringRef IndexRoot)
      : Index(Index) {
    llvm::SmallString<256> NativePath = IndexRoot;
    llvm::sys::path::native(NativePath);
    ProtobufMarshaller = std::unique_ptr<Marshaller>(new Marshaller(
        /*RemoteIndexRoot=*/llvm::StringRef(NativePath),
        /*LocalIndexRoot=*/""));
  }

private:
  using stopwatch = std::chrono::steady_clock;

  grpc::Status Lookup(grpc::ServerContext *Context,
                      const LookupRequest *Request,
                      grpc::ServerWriter<LookupReply> *Reply) override {
    auto StartTime = stopwatch::now();
    WithContextValue WithRequestContext(CurrentRequest, Context);
    logRequest(*Request);
    trace::Span Tracer("LookupRequest");
    auto Req = ProtobufMarshaller->fromProtobuf(Request);
    if (!Req) {
      elog("Can not parse LookupRequest from protobuf: {0}", Req.takeError());
      return grpc::Status::CANCELLED;
    }
    unsigned Sent = 0;
    unsigned FailedToSend = 0;
    bool HasMore = false;
    Index.lookup(*Req, [&](const clangd::Symbol &Item) {
      if (Sent >= LimitResults) {
        HasMore = true;
        return;
      }
      auto SerializedItem = ProtobufMarshaller->toProtobuf(Item);
      if (!SerializedItem) {
        elog("Unable to convert Symbol to protobuf: {0}",
             SerializedItem.takeError());
        ++FailedToSend;
        return;
      }
      LookupReply NextMessage;
      *NextMessage.mutable_stream_result() = *SerializedItem;
      logResponse(NextMessage);
      Reply->Write(NextMessage);
      ++Sent;
    });
    if (HasMore)
      log("[public] Limiting result size for Lookup request.");
    LookupReply LastMessage;
    LastMessage.mutable_final_result()->set_has_more(HasMore);
    logResponse(LastMessage);
    Reply->Write(LastMessage);
    SPAN_ATTACH(Tracer, "Sent", Sent);
    SPAN_ATTACH(Tracer, "Failed to send", FailedToSend);
    logRequestSummary("v1/Lookup", Sent, StartTime);
    return grpc::Status::OK;
  }

  grpc::Status FuzzyFind(grpc::ServerContext *Context,
                         const FuzzyFindRequest *Request,
                         grpc::ServerWriter<FuzzyFindReply> *Reply) override {
    auto StartTime = stopwatch::now();
    WithContextValue WithRequestContext(CurrentRequest, Context);
    logRequest(*Request);
    trace::Span Tracer("FuzzyFindRequest");
    auto Req = ProtobufMarshaller->fromProtobuf(Request);
    if (!Req) {
      elog("Can not parse FuzzyFindRequest from protobuf: {0}",
           Req.takeError());
      return grpc::Status::CANCELLED;
    }
    if (!Req->Limit || *Req->Limit > LimitResults) {
      log("[public] Limiting result size for FuzzyFind request from {0} to {1}",
          Req->Limit, LimitResults);
      Req->Limit = LimitResults;
    }
    unsigned Sent = 0;
    unsigned FailedToSend = 0;
    bool HasMore = Index.fuzzyFind(*Req, [&](const clangd::Symbol &Item) {
      auto SerializedItem = ProtobufMarshaller->toProtobuf(Item);
      if (!SerializedItem) {
        elog("Unable to convert Symbol to protobuf: {0}",
             SerializedItem.takeError());
        ++FailedToSend;
        return;
      }
      FuzzyFindReply NextMessage;
      *NextMessage.mutable_stream_result() = *SerializedItem;
      logResponse(NextMessage);
      Reply->Write(NextMessage);
      ++Sent;
    });
    FuzzyFindReply LastMessage;
    LastMessage.mutable_final_result()->set_has_more(HasMore);
    logResponse(LastMessage);
    Reply->Write(LastMessage);
    SPAN_ATTACH(Tracer, "Sent", Sent);
    SPAN_ATTACH(Tracer, "Failed to send", FailedToSend);
    logRequestSummary("v1/FuzzyFind", Sent, StartTime);
    return grpc::Status::OK;
  }

  grpc::Status Refs(grpc::ServerContext *Context, const RefsRequest *Request,
                    grpc::ServerWriter<RefsReply> *Reply) override {
    auto StartTime = stopwatch::now();
    WithContextValue WithRequestContext(CurrentRequest, Context);
    logRequest(*Request);
    trace::Span Tracer("RefsRequest");
    auto Req = ProtobufMarshaller->fromProtobuf(Request);
    if (!Req) {
      elog("Can not parse RefsRequest from protobuf: {0}", Req.takeError());
      return grpc::Status::CANCELLED;
    }
    if (!Req->Limit || *Req->Limit > LimitResults) {
      log("[public] Limiting result size for Refs request from {0} to {1}.",
          Req->Limit, LimitResults);
      Req->Limit = LimitResults;
    }
    unsigned Sent = 0;
    unsigned FailedToSend = 0;
    bool HasMore = Index.refs(*Req, [&](const clangd::Ref &Item) {
      auto SerializedItem = ProtobufMarshaller->toProtobuf(Item);
      if (!SerializedItem) {
        elog("Unable to convert Ref to protobuf: {0}",
             SerializedItem.takeError());
        ++FailedToSend;
        return;
      }
      RefsReply NextMessage;
      *NextMessage.mutable_stream_result() = *SerializedItem;
      logResponse(NextMessage);
      Reply->Write(NextMessage);
      ++Sent;
    });
    RefsReply LastMessage;
    LastMessage.mutable_final_result()->set_has_more(HasMore);
    logResponse(LastMessage);
    Reply->Write(LastMessage);
    SPAN_ATTACH(Tracer, "Sent", Sent);
    SPAN_ATTACH(Tracer, "Failed to send", FailedToSend);
    logRequestSummary("v1/Refs", Sent, StartTime);
    return grpc::Status::OK;
  }

  grpc::Status Relations(grpc::ServerContext *Context,
                         const RelationsRequest *Request,
                         grpc::ServerWriter<RelationsReply> *Reply) override {
    auto StartTime = stopwatch::now();
    WithContextValue WithRequestContext(CurrentRequest, Context);
    logRequest(*Request);
    trace::Span Tracer("RelationsRequest");
    auto Req = ProtobufMarshaller->fromProtobuf(Request);
    if (!Req) {
      elog("Can not parse RelationsRequest from protobuf: {0}",
           Req.takeError());
      return grpc::Status::CANCELLED;
    }
    if (!Req->Limit || *Req->Limit > LimitResults) {
      log("[public] Limiting result size for Relations request from {0} to "
          "{1}.",
          Req->Limit, LimitResults);
      Req->Limit = LimitResults;
    }
    unsigned Sent = 0;
    unsigned FailedToSend = 0;
    Index.relations(
        *Req, [&](const SymbolID &Subject, const clangd::Symbol &Object) {
          auto SerializedItem = ProtobufMarshaller->toProtobuf(Subject, Object);
          if (!SerializedItem) {
            elog("Unable to convert Relation to protobuf: {0}",
                 SerializedItem.takeError());
            ++FailedToSend;
            return;
          }
          RelationsReply NextMessage;
          *NextMessage.mutable_stream_result() = *SerializedItem;
          logResponse(NextMessage);
          Reply->Write(NextMessage);
          ++Sent;
        });
    RelationsReply LastMessage;
    LastMessage.mutable_final_result()->set_has_more(true);
    logResponse(LastMessage);
    Reply->Write(LastMessage);
    SPAN_ATTACH(Tracer, "Sent", Sent);
    SPAN_ATTACH(Tracer, "Failed to send", FailedToSend);
    logRequestSummary("v1/Relations", Sent, StartTime);
    return grpc::Status::OK;
  }

  // Proxy object to allow proto messages to be lazily serialized as text.
  struct TextProto {
    const google::protobuf::Message &M;
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                         const TextProto &P) {
      return OS << P.M.DebugString();
    }
  };

  void logRequest(const google::protobuf::Message &M) {
    vlog("<<< {0}\n{1}", M.GetDescriptor()->name(), TextProto{M});
  }
  void logResponse(const google::protobuf::Message &M) {
    vlog(">>> {0}\n{1}", M.GetDescriptor()->name(), TextProto{M});
  }
  void logRequestSummary(llvm::StringLiteral RequestName, unsigned Sent,
                         stopwatch::time_point StartTime) {
    auto Duration = stopwatch::now() - StartTime;
    auto Millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(Duration).count();
    log("[public] request {0} => OK: {1} results in {2}ms", RequestName, Sent,
        Millis);
  }

  std::unique_ptr<Marshaller> ProtobufMarshaller;
  clangd::SymbolIndex &Index;
};

class Monitor final : public v1::Monitor::Service {
public:
  Monitor(llvm::sys::TimePoint<> IndexAge)
      : StartTime(std::chrono::system_clock::now()), IndexBuildTime(IndexAge) {}

  void updateIndex(llvm::sys::TimePoint<> UpdateTime) {
    IndexBuildTime.exchange(UpdateTime);
  }

private:
  // FIXME(kirillbobyrev): Most fields should be populated when the index
  // reloads (probably in adjacent metadata.txt file next to loaded .idx) but
  // they aren't right now.
  grpc::Status MonitoringInfo(grpc::ServerContext *Context,
                              const v1::MonitoringInfoRequest *Request,
                              v1::MonitoringInfoReply *Reply) override {
    Reply->set_uptime_seconds(std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::system_clock::now() - StartTime)
                                  .count());
    // FIXME(kirillbobyrev): We are currently making use of the last
    // modification time of the index artifact to deduce its age. This is wrong
    // as it doesn't account for the indexing delay. Propagate some metadata
    // with the index artifacts to indicate time of the commit we indexed.
    Reply->set_index_age_seconds(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now() - IndexBuildTime.load())
            .count());
    return grpc::Status::OK;
  }

  const llvm::sys::TimePoint<> StartTime;
  std::atomic<llvm::sys::TimePoint<>> IndexBuildTime;
};

void maybeTrimMemory() {
#if defined(__GLIBC__) && CLANGD_MALLOC_TRIM
  malloc_trim(0);
#endif
}

// Detect changes in \p IndexPath file and load new versions of the index
// whenever they become available.
void hotReload(clangd::SwapIndex &Index, llvm::StringRef IndexPath,
               llvm::vfs::Status &LastStatus,
               llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> &FS,
               Monitor &Monitor) {
  // glibc malloc doesn't shrink an arena if there are items living at the end,
  // which might happen since we destroy the old index after building new one.
  // Trim more aggresively to keep memory usage of the server low.
  // Note that we do it deliberately here rather than after Index.reset(),
  // because old index might still be kept alive after the reset call if we are
  // serving requests.
  maybeTrimMemory();
  auto Status = FS->status(IndexPath);
  // Requested file is same as loaded index: no reload is needed.
  if (!Status || (Status->getLastModificationTime() ==
                      LastStatus.getLastModificationTime() &&
                  Status->getSize() == LastStatus.getSize()))
    return;
  vlog("Found different index version: existing index was modified at "
       "{0}, new index was modified at {1}. Attempting to reload.",
       LastStatus.getLastModificationTime(), Status->getLastModificationTime());
  LastStatus = *Status;
  std::unique_ptr<clang::clangd::SymbolIndex> NewIndex =
      loadIndex(IndexPath, SymbolOrigin::Static);
  if (!NewIndex) {
    elog("Failed to load new index. Old index will be served.");
    return;
  }
  Index.reset(std::move(NewIndex));
  Monitor.updateIndex(Status->getLastModificationTime());
  log("New index version loaded. Last modification time: {0}, size: {1} bytes.",
      Status->getLastModificationTime(), Status->getSize());
}

void runServerAndWait(clangd::SymbolIndex &Index, llvm::StringRef ServerAddress,
                      llvm::StringRef IndexPath, Monitor &Monitor) {
  RemoteIndexServer Service(Index, IndexRoot);

  grpc::EnableDefaultHealthCheckService(true);
#if ENABLE_GRPC_REFLECTION
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
#endif
  grpc::ServerBuilder Builder;
  Builder.AddListeningPort(ServerAddress.str(),
                           grpc::InsecureServerCredentials());
  Builder.AddChannelArgument(GRPC_ARG_MAX_CONNECTION_IDLE_MS,
                             IdleTimeoutSeconds * 1000);
  Builder.RegisterService(&Service);
  Builder.RegisterService(&Monitor);
  std::unique_ptr<grpc::Server> Server(Builder.BuildAndStart());
  log("Server listening on {0}", ServerAddress);

  std::thread ServerShutdownWatcher([&]() {
    static constexpr auto WatcherFrequency = std::chrono::seconds(5);
    while (!clang::clangd::shutdownRequested())
      std::this_thread::sleep_for(WatcherFrequency);
    Server->Shutdown();
  });

  Server->Wait();
  ServerShutdownWatcher.join();
}

std::unique_ptr<Logger> makeLogger(llvm::StringRef LogPrefix,
                                   llvm::raw_ostream &OS) {
  std::unique_ptr<Logger> Base;
  if (LogPublic) {
    // Redacted mode:
    //  - messages outside the scope of a request: log fully
    //  - messages tagged [public]: log fully
    //  - errors: log the format string
    //  - others: drop
    class RedactedLogger : public StreamLogger {
    public:
      using StreamLogger::StreamLogger;
      void log(Level L, const char *Fmt,
               const llvm::formatv_object_base &Message) override {
        if (Context::current().get(CurrentRequest) == nullptr ||
            llvm::StringRef(Fmt).startswith("[public]"))
          return StreamLogger::log(L, Fmt, Message);
        if (L >= Error)
          return StreamLogger::log(L, Fmt,
                                   llvm::formatv("[redacted] {0}", Fmt));
      }
    };
    Base = std::make_unique<RedactedLogger>(OS, LogLevel);
  } else {
    Base = std::make_unique<StreamLogger>(OS, LogLevel);
  }

  if (LogPrefix.empty())
    return Base;
  class PrefixedLogger : public Logger {
    std::string LogPrefix;
    std::unique_ptr<Logger> Base;

  public:
    PrefixedLogger(llvm::StringRef LogPrefix, std::unique_ptr<Logger> Base)
        : LogPrefix(LogPrefix.str()), Base(std::move(Base)) {}
    void log(Level L, const char *Fmt,
             const llvm::formatv_object_base &Message) override {
      Base->log(L, Fmt, llvm::formatv("[{0}] {1}", LogPrefix, Message));
    }
  };
  return std::make_unique<PrefixedLogger>(LogPrefix, std::move(Base));
}

} // namespace
} // namespace remote
} // namespace clangd
} // namespace clang

using clang::clangd::elog;

int main(int argc, char *argv[]) {
  using namespace clang::clangd::remote;
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::sys::SetInterruptFunction(&clang::clangd::requestShutdown);

  if (!llvm::sys::path::is_absolute(IndexRoot)) {
    llvm::errs() << "Index root should be an absolute path.\n";
    return -1;
  }

  llvm::errs().SetBuffered();
  // Don't flush stdout when logging for thread safety.
  llvm::errs().tie(nullptr);
  auto Logger = makeLogger(LogPrefix.getValue(), llvm::errs());
  clang::clangd::LoggingSession LoggingSession(*Logger);

  std::optional<llvm::raw_fd_ostream> TracerStream;
  std::unique_ptr<clang::clangd::trace::EventTracer> Tracer;
  if (!TraceFile.empty()) {
    std::error_code EC;
    TracerStream.emplace(TraceFile, EC,
                         llvm::sys::fs::FA_Read | llvm::sys::fs::FA_Write);
    if (EC) {
      TracerStream.reset();
      elog("Error while opening trace file {0}: {1}", TraceFile, EC.message());
    } else {
      // FIXME(kirillbobyrev): Also create metrics tracer to track latency and
      // accumulate other request statistics.
      Tracer = clang::clangd::trace::createJSONTracer(*TracerStream,
                                                      /*PrettyPrint=*/false);
      clang::clangd::vlog("Successfully created a tracer.");
    }
  }

  std::optional<clang::clangd::trace::Session> TracingSession;
  if (Tracer)
    TracingSession.emplace(*Tracer);

  clang::clangd::RealThreadsafeFS TFS;
  auto FS = TFS.view(std::nullopt);
  auto Status = FS->status(IndexPath);
  if (!Status) {
    elog("{0} does not exist.", IndexPath);
    return Status.getError().value();
  }

  auto SymIndex =
      clang::clangd::loadIndex(IndexPath, clang::clangd::SymbolOrigin::Static);
  if (!SymIndex) {
    llvm::errs() << "Failed to open the index.\n";
    return -1;
  }
  clang::clangd::SwapIndex Index(std::move(SymIndex));

  Monitor Monitor(Status->getLastModificationTime());

  std::thread HotReloadThread([&Index, &Status, &FS, &Monitor]() {
    llvm::vfs::Status LastStatus = *Status;
    static constexpr auto RefreshFrequency = std::chrono::seconds(30);
    while (!clang::clangd::shutdownRequested()) {
      hotReload(Index, llvm::StringRef(IndexPath), LastStatus, FS, Monitor);
      std::this_thread::sleep_for(RefreshFrequency);
    }
  });

  runServerAndWait(Index, ServerAddress, IndexPath, Monitor);

  HotReloadThread.join();
}
