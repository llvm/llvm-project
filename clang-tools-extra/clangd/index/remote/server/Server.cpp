//===--- Server.cpp - gRPC-based Remote Index Server  ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/Index.h"
#include "index/Serialization.h"
#include "index/remote/marshalling/Marshalling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
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

llvm::cl::opt<std::string> ServerAddress(
    "server-address", llvm::cl::init("0.0.0.0:50051"),
    llvm::cl::desc("Address of the invoked server. Defaults to 0.0.0.0:50051"));

std::unique_ptr<clangd::SymbolIndex> openIndex(llvm::StringRef Index) {
  return loadIndex(Index, /*UseIndex=*/true);
}

class RemoteIndexServer final : public SymbolIndex::Service {
public:
  RemoteIndexServer(std::unique_ptr<clangd::SymbolIndex> Index)
      : Index(std::move(Index)) {}

private:
  grpc::Status Lookup(grpc::ServerContext *Context,
                      const LookupRequest *Request,
                      grpc::ServerWriter<LookupReply> *Reply) override {
    clangd::LookupRequest Req;
    for (const auto &ID : Request->ids()) {
      auto SID = SymbolID::fromStr(StringRef(ID));
      if (!SID)
        return grpc::Status::CANCELLED;
      Req.IDs.insert(*SID);
    }
    Index->lookup(Req, [&](const clangd::Symbol &Sym) {
      LookupReply NextMessage;
      *NextMessage.mutable_stream_result() = toProtobuf(Sym);
      Reply->Write(NextMessage);
    });
    LookupReply LastMessage;
    LastMessage.set_final_result(true);
    Reply->Write(LastMessage);
    return grpc::Status::OK;
  }

  grpc::Status FuzzyFind(grpc::ServerContext *Context,
                         const FuzzyFindRequest *Request,
                         grpc::ServerWriter<FuzzyFindReply> *Reply) override {
    const auto Req = fromProtobuf(Request);
    bool HasMore = Index->fuzzyFind(Req, [&](const clangd::Symbol &Sym) {
      FuzzyFindReply NextMessage;
      *NextMessage.mutable_stream_result() = toProtobuf(Sym);
      Reply->Write(NextMessage);
    });
    FuzzyFindReply LastMessage;
    LastMessage.set_final_result(HasMore);
    Reply->Write(LastMessage);
    return grpc::Status::OK;
  }

  grpc::Status Refs(grpc::ServerContext *Context, const RefsRequest *Request,
                    grpc::ServerWriter<RefsReply> *Reply) override {
    clangd::RefsRequest Req;
    for (const auto &ID : Request->ids()) {
      auto SID = SymbolID::fromStr(StringRef(ID));
      if (!SID)
        return grpc::Status::CANCELLED;
      Req.IDs.insert(*SID);
    }
    bool HasMore = Index->refs(Req, [&](const clangd::Ref &Reference) {
      RefsReply NextMessage;
      *NextMessage.mutable_stream_result() = toProtobuf(Reference);
      Reply->Write(NextMessage);
    });
    RefsReply LastMessage;
    LastMessage.set_final_result(HasMore);
    Reply->Write(LastMessage);
    return grpc::Status::OK;
  }

  std::unique_ptr<clangd::SymbolIndex> Index;
};

void runServer(std::unique_ptr<clangd::SymbolIndex> Index,
               const std::string &ServerAddress) {
  RemoteIndexServer Service(std::move(Index));

  grpc::EnableDefaultHealthCheckService(true);
  grpc::ServerBuilder Builder;
  Builder.AddListeningPort(ServerAddress, grpc::InsecureServerCredentials());
  Builder.RegisterService(&Service);
  std::unique_ptr<grpc::Server> Server(Builder.BuildAndStart());
  llvm::outs() << "Server listening on " << ServerAddress << '\n';

  Server->Wait();
}

} // namespace
} // namespace remote
} // namespace clangd
} // namespace clang

int main(int argc, char *argv[]) {
  using namespace clang::clangd::remote;
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  std::unique_ptr<clang::clangd::SymbolIndex> Index = openIndex(IndexPath);

  if (!Index) {
    llvm::outs() << "Failed to open the index.\n";
    return -1;
  }

  runServer(std::move(Index), ServerAddress);
}
