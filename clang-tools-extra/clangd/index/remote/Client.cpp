//===--- Client.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <grpc++/grpc++.h>

#include "Client.h"
#include "Index.grpc.pb.h"
#include "index/Serialization.h"
#include "marshalling/Marshalling.h"
#include "support/Logger.h"
#include "support/Trace.h"

namespace clang {
namespace clangd {
namespace remote {
namespace {

class IndexClient : public clangd::SymbolIndex {
  template <typename RequestT, typename ReplyT>
  using StreamingCall = std::unique_ptr<grpc::ClientReader<ReplyT>> (
      remote::SymbolIndex::Stub::*)(grpc::ClientContext *, const RequestT &);

  // FIXME(kirillbobyrev): Set deadlines for requests.
  template <typename RequestT, typename ReplyT, typename ClangdRequestT,
            typename CallbackT>
  bool streamRPC(ClangdRequestT Request,
                 StreamingCall<RequestT, ReplyT> RPCCall,
                 CallbackT Callback) const {
    bool FinalResult = false;
    trace::Span Tracer(RequestT::descriptor()->name());
    const auto RPCRequest = toProtobuf(Request);
    grpc::ClientContext Context;
    auto Reader = (Stub.get()->*RPCCall)(&Context, RPCRequest);
    llvm::BumpPtrAllocator Arena;
    llvm::UniqueStringSaver Strings(Arena);
    ReplyT Reply;
    while (Reader->Read(&Reply)) {
      if (!Reply.has_stream_result()) {
        FinalResult = Reply.final_result();
        continue;
      }
      auto Sym = fromProtobuf(Reply.stream_result(), &Strings);
      if (!Sym)
        elog("Received invalid {0}", ReplyT::descriptor()->name());
      Callback(*Sym);
    }
    SPAN_ATTACH(Tracer, "status", Reader->Finish().ok());
    return FinalResult;
  }

public:
  IndexClient(std::shared_ptr<grpc::Channel> Channel)
      : Stub(remote::SymbolIndex::NewStub(Channel)) {}

  void lookup(const clangd::LookupRequest &Request,
              llvm::function_ref<void(const clangd::Symbol &)> Callback) const {
    streamRPC(Request, &remote::SymbolIndex::Stub::Lookup, Callback);
  }

  bool
  fuzzyFind(const clangd::FuzzyFindRequest &Request,
            llvm::function_ref<void(const clangd::Symbol &)> Callback) const {
    return streamRPC(Request, &remote::SymbolIndex::Stub::FuzzyFind, Callback);
  }

  bool refs(const clangd::RefsRequest &Request,
            llvm::function_ref<void(const clangd::Ref &)> Callback) const {
    return streamRPC(Request, &remote::SymbolIndex::Stub::Refs, Callback);
  }

  // FIXME(kirillbobyrev): Implement this.
  void
  relations(const clangd::RelationsRequest &,
            llvm::function_ref<void(const SymbolID &, const clangd::Symbol &)>)
      const {}

  // IndexClient does not take any space since the data is stored on the server.
  size_t estimateMemoryUsage() const { return 0; }

private:
  std::unique_ptr<remote::SymbolIndex::Stub> Stub;
};

} // namespace

std::unique_ptr<clangd::SymbolIndex> getClient(llvm::StringRef Address) {
  const auto Channel =
      grpc::CreateChannel(Address.str(), grpc::InsecureChannelCredentials());
  Channel->GetState(true);
  return std::unique_ptr<clangd::SymbolIndex>(new IndexClient(Channel));
}

} // namespace remote
} // namespace clangd
} // namespace clang
