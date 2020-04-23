//===--- Client.cpp - Remote Index Client -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple interactive tool which can be used to manually
// evaluate symbol search quality of Clangd index.
//
//===----------------------------------------------------------------------===//

#include "SourceCode.h"
#include "index/Serialization.h"
#include "index/dex/Dex.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "grpcpp/grpcpp.h"

#include "Index.grpc.pb.h"

namespace clang {
namespace clangd {
namespace {

llvm::cl::opt<std::string>
    ServerAddress("server-address",
                  llvm::cl::desc("Address of remote index server to use."),
                  llvm::cl::init("0.0.0.0:50051"));

static const std::string Overview = R"(
This is an **experimental** interactive tool to process user-provided search
queries over given symbol collection obtained via clangd-indexer with the help
of remote index server. The client will connect to remote index server and pass
it lookup queries.
)";

class RemoteIndexClient {
public:
  RemoteIndexClient(std::shared_ptr<grpc::Channel> Channel)
      : Stub(remote::Index::NewStub(Channel)) {}

  void lookup(llvm::StringRef ID) {
    llvm::outs() << "Lookup of symbol with ID " << ID << '\n';
    remote::LookupRequest Proto;
    Proto.set_id(ID.str());

    grpc::ClientContext Context;
    remote::LookupReply Reply;
    std::unique_ptr<grpc::ClientReader<remote::LookupReply>> Reader(
        Stub->Lookup(&Context, Proto));
    while (Reader->Read(&Reply)) {
      llvm::outs() << Reply.symbol_yaml();
    }
    grpc::Status Status = Reader->Finish();
    if (Status.ok()) {
      llvm::outs() << "lookupRequest rpc succeeded.\n";
    } else {
      llvm::outs() << "lookupRequest rpc failed.\n";
    }
  }

private:
  std::unique_ptr<remote::Index::Stub> Stub;
};

} // namespace
} // namespace clangd
} // namespace clang

int main(int argc, const char *argv[]) {
  using namespace clang::clangd;

  llvm::cl::ParseCommandLineOptions(argc, argv, Overview);
  llvm::cl::ResetCommandLineParser(); // We reuse it for REPL commands.
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  RemoteIndexClient IndexClient(
      grpc::CreateChannel(ServerAddress, grpc::InsecureChannelCredentials()));

  llvm::LineEditor LE("remote-index-client");
  while (llvm::Optional<std::string> Request = LE.readLine())
    IndexClient.lookup(std::move(*Request));
}
