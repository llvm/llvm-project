//===--- Server.cpp - gRPC-based Remote Index Server  ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/Index.h"
#include "index/Serialization.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"

#include "grpcpp/grpcpp.h"
#include "grpcpp/health_check_service_interface.h"

#include "Index.grpc.pb.h"

namespace clang {
namespace clangd {
namespace {

static const std::string Overview = R"(
This is an experimental remote index implementation. The server opens Dex and
awaits gRPC lookup requests from the client.
)";

llvm::cl::opt<std::string> IndexPath(llvm::cl::desc("<INDEX FILE>"),
                                     llvm::cl::Positional, llvm::cl::Required);

llvm::cl::opt<std::string> ServerAddress("server-address",
                                         llvm::cl::init("0.0.0.0:50051"));

std::unique_ptr<SymbolIndex> openIndex(llvm::StringRef Index) {
  return loadIndex(Index, /*UseIndex=*/true);
}

class RemoteIndexServer final : public remote::Index::Service {
public:
  RemoteIndexServer(std::unique_ptr<SymbolIndex> Index)
      : Index(std::move(Index)) {}

private:
  grpc::Status Lookup(grpc::ServerContext *Context,
                      const remote::LookupRequest *Request,
                      grpc::ServerWriter<remote::LookupReply> *Reply) override {
    llvm::outs() << "Lookup of symbol with ID " << Request->id() << '\n';
    LookupRequest Req;
    auto SID = SymbolID::fromStr(Request->id());
    if (!SID) {
      llvm::outs() << llvm::toString(SID.takeError()) << "\n";
      return grpc::Status::CANCELLED;
    }
    Req.IDs.insert(*SID);
    Index->lookup(Req, [&](const Symbol &Sym) {
      remote::LookupReply NextSymbol;
      NextSymbol.set_symbol_yaml(toYAML(Sym));
      Reply->Write(NextSymbol);
    });
    return grpc::Status::OK;
  }

  std::unique_ptr<SymbolIndex> Index;
};

void runServer(std::unique_ptr<SymbolIndex> Index,
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
} // namespace clangd
} // namespace clang

int main(int argc, char *argv[]) {
  using namespace clang::clangd;
  llvm::cl::ParseCommandLineOptions(argc, argv, clang::clangd::Overview);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  std::unique_ptr<SymbolIndex> Index = openIndex(IndexPath);

  if (!Index) {
    llvm::outs() << "Failed to open the index.\n";
    return -1;
  }

  runServer(std::move(Index), ServerAddress);
}
