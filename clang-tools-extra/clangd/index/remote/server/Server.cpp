//===--- Server.cpp - gRPC-based Remote Index Server  ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Index.pb.h"
#include "index/Index.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/remote/marshalling/Marshalling.h"
#include "support/Logger.h"
#include "support/Shutdown.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <chrono>
#include <grpc++/grpc++.h>
#include <grpc++/health_check_service_interface.h>
#include <memory>
#include <thread>

#include "Index.grpc.pb.h"

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

class RemoteIndexServer final : public SymbolIndex::Service {
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
  grpc::Status Lookup(grpc::ServerContext *Context,
                      const LookupRequest *Request,
                      grpc::ServerWriter<LookupReply> *Reply) override {
    trace::Span Tracer("LookupRequest");
    auto Req = ProtobufMarshaller->fromProtobuf(Request);
    if (!Req) {
      elog("Can not parse LookupRequest from protobuf: {0}", Req.takeError());
      return grpc::Status::CANCELLED;
    }
    unsigned Sent = 0;
    unsigned FailedToSend = 0;
    Index.lookup(*Req, [&](const clangd::Symbol &Item) {
      auto SerializedItem = ProtobufMarshaller->toProtobuf(Item);
      if (!SerializedItem) {
        elog("Unable to convert Symbol to protobuf: {0}",
             SerializedItem.takeError());
        ++FailedToSend;
        return;
      }
      LookupReply NextMessage;
      *NextMessage.mutable_stream_result() = *SerializedItem;
      Reply->Write(NextMessage);
      ++Sent;
    });
    LookupReply LastMessage;
    LastMessage.set_final_result(true);
    Reply->Write(LastMessage);
    SPAN_ATTACH(Tracer, "Sent", Sent);
    SPAN_ATTACH(Tracer, "Failed to send", FailedToSend);
    return grpc::Status::OK;
  }

  grpc::Status FuzzyFind(grpc::ServerContext *Context,
                         const FuzzyFindRequest *Request,
                         grpc::ServerWriter<FuzzyFindReply> *Reply) override {
    trace::Span Tracer("FuzzyFindRequest");
    auto Req = ProtobufMarshaller->fromProtobuf(Request);
    if (!Req) {
      elog("Can not parse FuzzyFindRequest from protobuf: {0}",
           Req.takeError());
      return grpc::Status::CANCELLED;
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
      Reply->Write(NextMessage);
      ++Sent;
    });
    FuzzyFindReply LastMessage;
    LastMessage.set_final_result(HasMore);
    Reply->Write(LastMessage);
    SPAN_ATTACH(Tracer, "Sent", Sent);
    SPAN_ATTACH(Tracer, "Failed to send", FailedToSend);
    return grpc::Status::OK;
  }

  grpc::Status Refs(grpc::ServerContext *Context, const RefsRequest *Request,
                    grpc::ServerWriter<RefsReply> *Reply) override {
    trace::Span Tracer("RefsRequest");
    auto Req = ProtobufMarshaller->fromProtobuf(Request);
    if (!Req) {
      elog("Can not parse RefsRequest from protobuf: {0}", Req.takeError());
      return grpc::Status::CANCELLED;
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
      Reply->Write(NextMessage);
      ++Sent;
    });
    RefsReply LastMessage;
    LastMessage.set_final_result(HasMore);
    Reply->Write(LastMessage);
    SPAN_ATTACH(Tracer, "Sent", Sent);
    SPAN_ATTACH(Tracer, "Failed to send", FailedToSend);
    return grpc::Status::OK;
  }

  grpc::Status Relations(grpc::ServerContext *Context,
                         const RelationsRequest *Request,
                         grpc::ServerWriter<RelationsReply> *Reply) override {
    trace::Span Tracer("RelationsRequest");
    auto Req = ProtobufMarshaller->fromProtobuf(Request);
    if (!Req) {
      elog("Can not parse RelationsRequest from protobuf: {0}",
           Req.takeError());
      return grpc::Status::CANCELLED;
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
          Reply->Write(NextMessage);
          ++Sent;
        });
    RelationsReply LastMessage;
    LastMessage.set_final_result(true);
    Reply->Write(LastMessage);
    SPAN_ATTACH(Tracer, "Sent", Sent);
    SPAN_ATTACH(Tracer, "Failed to send", FailedToSend);
    return grpc::Status::OK;
  }

  std::unique_ptr<Marshaller> ProtobufMarshaller;
  clangd::SymbolIndex &Index;
};

// Detect changes in \p IndexPath file and load new versions of the index
// whenever they become available.
void hotReload(clangd::SwapIndex &Index, llvm::StringRef IndexPath,
               llvm::vfs::Status &LastStatus,
               llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> &FS) {
  auto Status = FS->status(IndexPath);
  // Requested file is same as loaded index: no reload is needed.
  if (!Status || (Status->getLastModificationTime() ==
                      LastStatus.getLastModificationTime() &&
                  Status->getSize() == LastStatus.getSize()))
    return;
  vlog("Found different index version: existing index was modified at {0}, new "
       "index was modified at {1}. Attempting to reload.",
       LastStatus.getLastModificationTime(), Status->getLastModificationTime());
  LastStatus = *Status;
  std::unique_ptr<clang::clangd::SymbolIndex> NewIndex = loadIndex(IndexPath);
  if (!NewIndex) {
    elog("Failed to load new index. Old index will be served.");
    return;
  }
  Index.reset(std::move(NewIndex));
  log("New index version loaded. Last modification time: {0}, size: {1} bytes.",
      Status->getLastModificationTime(), Status->getSize());
}

void runServerAndWait(clangd::SymbolIndex &Index, llvm::StringRef ServerAddress,
                      llvm::StringRef IndexPath) {
  RemoteIndexServer Service(Index, IndexRoot);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::ServerBuilder Builder;
  Builder.AddListeningPort(ServerAddress.str(),
                           grpc::InsecureServerCredentials());
  Builder.RegisterService(&Service);
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
  clang::clangd::StreamLogger Logger(llvm::errs(), LogLevel);
  clang::clangd::LoggingSession LoggingSession(Logger);

  llvm::Optional<llvm::raw_fd_ostream> TracerStream;
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

  llvm::Optional<clang::clangd::trace::Session> TracingSession;
  if (Tracer)
    TracingSession.emplace(*Tracer);

  clang::clangd::RealThreadsafeFS TFS;
  auto FS = TFS.view(llvm::None);
  auto Status = FS->status(IndexPath);
  if (!Status) {
    elog("{0} does not exist.", IndexPath);
    return Status.getError().value();
  }

  auto Index = std::make_unique<clang::clangd::SwapIndex>(
      clang::clangd::loadIndex(IndexPath));

  if (!Index) {
    llvm::errs() << "Failed to open the index.\n";
    return -1;
  }

  std::thread HotReloadThread([&Index, &Status, &FS]() {
    llvm::vfs::Status LastStatus = *Status;
    static constexpr auto RefreshFrequency = std::chrono::seconds(90);
    while (!clang::clangd::shutdownRequested()) {
      hotReload(*Index, llvm::StringRef(IndexPath), LastStatus, FS);
      std::this_thread::sleep_for(RefreshFrequency);
    }
  });

  runServerAndWait(*Index, ServerAddress, IndexPath);

  HotReloadThread.join();
}
