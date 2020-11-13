//===--- Client.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <grpc++/grpc++.h>

#include "Client.h"
#include "Service.grpc.pb.h"
#include "index/Index.h"
#include "marshalling/Marshalling.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <chrono>

namespace clang {
namespace clangd {
namespace remote {
namespace {

class IndexClient : public clangd::SymbolIndex {
  template <typename RequestT, typename ReplyT>
  using StreamingCall = std::unique_ptr<grpc::ClientReader<ReplyT>> (
      remote::v1::SymbolIndex::Stub::*)(grpc::ClientContext *,
                                        const RequestT &);

  template <typename RequestT, typename ReplyT, typename ClangdRequestT,
            typename CallbackT>
  bool streamRPC(ClangdRequestT Request,
                 StreamingCall<RequestT, ReplyT> RPCCall,
                 CallbackT Callback) const {
    bool FinalResult = false;
    trace::Span Tracer(RequestT::descriptor()->name());
    const auto RPCRequest = ProtobufMarshaller->toProtobuf(Request);
    SPAN_ATTACH(Tracer, "Request", RPCRequest.DebugString());
    grpc::ClientContext Context;
    Context.AddMetadata("version", clang::getClangToolFullVersion("clangd"));
    std::chrono::system_clock::time_point Deadline =
        std::chrono::system_clock::now() + DeadlineWaitingTime;
    Context.set_deadline(Deadline);
    auto Reader = (Stub.get()->*RPCCall)(&Context, RPCRequest);
    ReplyT Reply;
    unsigned Successful = 0;
    unsigned FailedToParse = 0;
    while (Reader->Read(&Reply)) {
      if (!Reply.has_stream_result()) {
        FinalResult = Reply.final_result().has_more();
        continue;
      }
      auto Response = ProtobufMarshaller->fromProtobuf(Reply.stream_result());
      if (!Response) {
        elog("Received invalid {0}: {1}. Reason: {2}",
             ReplyT::descriptor()->name(), Reply.stream_result().DebugString(),
             Response.takeError());
        ++FailedToParse;
        continue;
      }
      Callback(*Response);
      ++Successful;
    }
    SPAN_ATTACH(Tracer, "Status", Reader->Finish().ok());
    SPAN_ATTACH(Tracer, "Successful", Successful);
    SPAN_ATTACH(Tracer, "Failed to parse", FailedToParse);
    return FinalResult;
  }

public:
  IndexClient(
      std::shared_ptr<grpc::Channel> Channel, llvm::StringRef ProjectRoot,
      std::chrono::milliseconds DeadlineTime = std::chrono::milliseconds(1000))
      : Stub(remote::v1::SymbolIndex::NewStub(Channel)),
        ProtobufMarshaller(new Marshaller(/*RemoteIndexRoot=*/"",
                                          /*LocalIndexRoot=*/ProjectRoot)),
        DeadlineWaitingTime(DeadlineTime) {
    assert(!ProjectRoot.empty());
  }

  void lookup(const clangd::LookupRequest &Request,
              llvm::function_ref<void(const clangd::Symbol &)> Callback)
      const override {
    streamRPC(Request, &remote::v1::SymbolIndex::Stub::Lookup, Callback);
  }

  bool fuzzyFind(const clangd::FuzzyFindRequest &Request,
                 llvm::function_ref<void(const clangd::Symbol &)> Callback)
      const override {
    return streamRPC(Request, &remote::v1::SymbolIndex::Stub::FuzzyFind,
                     Callback);
  }

  bool
  refs(const clangd::RefsRequest &Request,
       llvm::function_ref<void(const clangd::Ref &)> Callback) const override {
    return streamRPC(Request, &remote::v1::SymbolIndex::Stub::Refs, Callback);
  }

  void
  relations(const clangd::RelationsRequest &Request,
            llvm::function_ref<void(const SymbolID &, const clangd::Symbol &)>
                Callback) const override {
    streamRPC(Request, &remote::v1::SymbolIndex::Stub::Relations,
              // Unpack protobuf Relation.
              [&](std::pair<SymbolID, clangd::Symbol> SubjectAndObject) {
                Callback(SubjectAndObject.first, SubjectAndObject.second);
              });
  }

  // IndexClient does not take any space since the data is stored on the
  // server.
  size_t estimateMemoryUsage() const override { return 0; }

private:
  std::unique_ptr<remote::v1::SymbolIndex::Stub> Stub;
  std::unique_ptr<Marshaller> ProtobufMarshaller;
  // Each request will be terminated if it takes too long.
  std::chrono::milliseconds DeadlineWaitingTime;
};

} // namespace

std::unique_ptr<clangd::SymbolIndex> getClient(llvm::StringRef Address,
                                               llvm::StringRef ProjectRoot) {
  const auto Channel =
      grpc::CreateChannel(Address.str(), grpc::InsecureChannelCredentials());
  Channel->GetState(true);
  return std::unique_ptr<clangd::SymbolIndex>(
      new IndexClient(Channel, ProjectRoot));
}

} // namespace remote
} // namespace clangd
} // namespace clang
