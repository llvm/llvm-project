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
#include "support/Trace.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"

#include <grpc++/grpc++.h>
#include <grpc++/health_check_service_interface.h>

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

std::unique_ptr<clangd::SymbolIndex> openIndex(llvm::StringRef Index) {
  return loadIndex(Index, /*UseIndex=*/true);
}

class RemoteIndexServer final : public SymbolIndex::Service {
public:
  RemoteIndexServer(std::unique_ptr<clangd::SymbolIndex> Index,
                    llvm::StringRef IndexRoot)
      : Index(std::move(Index)) {
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
    Index->lookup(*Req, [&](const clangd::Symbol &Item) {
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
    bool HasMore = Index->fuzzyFind(*Req, [&](const clangd::Symbol &Item) {
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
    bool HasMore = Index->refs(*Req, [&](const clangd::Ref &Item) {
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
    Index->relations(
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

  std::unique_ptr<clangd::SymbolIndex> Index;
  std::unique_ptr<Marshaller> ProtobufMarshaller;
};

void runServer(std::unique_ptr<clangd::SymbolIndex> Index,
               const std::string &ServerAddress) {
  RemoteIndexServer Service(std::move(Index), IndexRoot);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::ServerBuilder Builder;
  Builder.AddListeningPort(ServerAddress, grpc::InsecureServerCredentials());
  Builder.RegisterService(&Service);
  std::unique_ptr<grpc::Server> Server(Builder.BuildAndStart());
  log("Server listening on {0}", ServerAddress);

  Server->Wait();
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

  std::unique_ptr<clang::clangd::SymbolIndex> Index = openIndex(IndexPath);

  if (!Index) {
    llvm::errs() << "Failed to open the index.\n";
    return -1;
  }

  runServer(std::move(Index), ServerAddress);
}
