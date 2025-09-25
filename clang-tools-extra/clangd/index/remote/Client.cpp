//===--- Client.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <grpc++/grpc++.h>

#include "Client.h"
#include "Feature.h"
#include "Service.grpc.pb.h"
#include "index/Index.h"
#include "marshalling/Marshalling.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <atomic>
#include <chrono>
#include <memory>

namespace clang {
namespace clangd {
namespace remote {
namespace {

llvm::StringRef toString(const grpc_connectivity_state &State) {
  switch (State) {
  case GRPC_CHANNEL_IDLE:
    return "idle";
  case GRPC_CHANNEL_CONNECTING:
    return "connecting";
  case GRPC_CHANNEL_READY:
    return "ready";
  case GRPC_CHANNEL_TRANSIENT_FAILURE:
    return "transient failure";
  case GRPC_CHANNEL_SHUTDOWN:
    return "shutdown";
  }
  llvm_unreachable("Not a valid grpc_connectivity_state.");
}

class IndexClient : public clangd::SymbolIndex {
  void updateConnectionStatus() const {
    auto NewStatus = Channel->GetState(/*try_to_connect=*/false);
    auto OldStatus = ConnectionStatus.exchange(NewStatus);
    if (OldStatus != NewStatus)
      vlog("Remote index connection [{0}]: {1} => {2}", ServerAddress,
           toString(OldStatus), toString(NewStatus));
  }

  template <typename RequestT, typename ReplyT>
  using StreamingCall = std::unique_ptr<grpc::ClientReader<ReplyT>> (
      remote::v1::SymbolIndex::Stub::*)(grpc::ClientContext *,
                                        const RequestT &);

  template <typename RequestT, typename ReplyT, typename ClangdRequestT,
            typename CallbackT>
  bool streamRPC(ClangdRequestT Request,
                 StreamingCall<RequestT, ReplyT> RPCCall,
                 CallbackT Callback) const {
    updateConnectionStatus();
    // We initialize to true because stream might be broken before we see the
    // final message. In such a case there are actually more results on the
    // stream, but we couldn't get to them.
    bool HasMore = true;
    trace::Span Tracer(RequestT::descriptor()->name());
    const auto RPCRequest = ProtobufMarshaller->toProtobuf(Request);
    SPAN_ATTACH(Tracer, "Request", RPCRequest.DebugString());
    grpc::ClientContext Context;
    Context.AddMetadata("version", versionString());
    Context.AddMetadata("features", featureString());
    Context.AddMetadata("platform", platformString());
    std::chrono::system_clock::time_point StartTime =
        std::chrono::system_clock::now();
    auto Deadline = StartTime + DeadlineWaitingTime;
    Context.set_deadline(Deadline);
    auto Reader = (Stub.get()->*RPCCall)(&Context, RPCRequest);
    dlog("Sending {0}: {1}", RequestT::descriptor()->name(),
         RPCRequest.DebugString());
    ReplyT Reply;
    unsigned Successful = 0;
    unsigned FailedToParse = 0;
    while (Reader->Read(&Reply)) {
      if (!Reply.has_stream_result()) {
        HasMore = Reply.final_result().has_more();
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
    auto Millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now() - StartTime)
                      .count();
    vlog("Remote index [{0}]: {1} => {2} results in {3}ms.", ServerAddress,
         RequestT::descriptor()->name(), Successful, Millis);
    SPAN_ATTACH(Tracer, "Status", Reader->Finish().ok());
    SPAN_ATTACH(Tracer, "Successful", Successful);
    SPAN_ATTACH(Tracer, "Failed to parse", FailedToParse);
    updateConnectionStatus();
    return HasMore;
  }

public:
  IndexClient(
      std::shared_ptr<grpc::Channel> Channel, llvm::StringRef Address,
      llvm::StringRef ProjectRoot,
      std::chrono::milliseconds DeadlineTime = std::chrono::milliseconds(1000))
      : Stub(remote::v1::SymbolIndex::NewStub(Channel)), Channel(Channel),
        ServerAddress(Address),
        ConnectionStatus(Channel->GetState(/*try_to_connect=*/true)),
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

  bool containedRefs(const clangd::ContainedRefsRequest &Request,
                     llvm::function_ref<void(const ContainedRefsResult &)>
                         Callback) const override {
    return streamRPC(Request, &remote::v1::SymbolIndex::Stub::ContainedRefs,
                     Callback);
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

  llvm::unique_function<IndexContents(llvm::StringRef) const>
  indexedFiles() const override {
    // FIXME: For now we always return IndexContents::None regardless of whether
    //        the file was indexed or not. A possible implementation could be
    //        based on the idea that we do not want to send a request at every
    //        call of a function returned by IndexClient::indexedFiles().
    return [](llvm::StringRef) { return IndexContents::None; };
  }

  // IndexClient does not take any space since the data is stored on the
  // server.
  size_t estimateMemoryUsage() const override { return 0; }

private:
  std::unique_ptr<remote::v1::SymbolIndex::Stub> Stub;
  std::shared_ptr<grpc::Channel> Channel;
  llvm::SmallString<256> ServerAddress;
  mutable std::atomic<grpc_connectivity_state> ConnectionStatus;
  std::unique_ptr<Marshaller> ProtobufMarshaller;
  // Each request will be terminated if it takes too long.
  std::chrono::milliseconds DeadlineWaitingTime;
};

} // namespace

std::unique_ptr<clangd::SymbolIndex> getClient(llvm::StringRef Address,
                                               llvm::StringRef ProjectRoot) {
  const auto Channel =
      grpc::CreateChannel(Address.str(), grpc::InsecureChannelCredentials());
  return std::unique_ptr<clangd::SymbolIndex>(
      new IndexClient(Channel, Address, ProjectRoot));
}

} // namespace remote
} // namespace clangd
} // namespace clang
