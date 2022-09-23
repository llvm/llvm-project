//===-- Client.cpp - Remote Client ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Client.h"
#include "compilation_caching_cas.grpc.pb.h"
#include "compilation_caching_kv.grpc.pb.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <grpcpp/grpcpp.h>

using namespace clang;
using namespace clang::remote_cache;
using namespace compilation_cache_service::cas::v1;
using namespace compilation_cache_service::keyvalue::v1;
using namespace llvm;

void AsyncCallerContext::anchor() {}
void AsyncQueueBase::anchor() {}

void KeyValueDBClient::GetValueAsyncQueue::anchor() {}
void KeyValueDBClient::PutValueAsyncQueue::anchor() {}
void KeyValueDBClient::anchor() {}

void CASDBClient::LoadAsyncQueue::anchor() {}
void CASDBClient::SaveAsyncQueue::anchor() {}
void CASDBClient::anchor() {}

static Error errorFromGRPCStatus(const grpc::Status &Status) {
  return createStringError(inconvertibleErrorCode(), Status.error_message());
}

namespace {

template <typename ResponseT> struct AsyncClientCall {
  std::shared_ptr<AsyncCallerContext> CallCtx;

  // Container for the data we expect from the server.
  ResponseT Response;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  grpc::ClientContext Context;

  // Storage for the status of the RPC upon completion.
  grpc::Status Status;

  std::unique_ptr<grpc::ClientAsyncResponseReader<ResponseT>> Reader;
};

class KVGetAsyncQueueImpl : public KeyValueDBClient::GetValueAsyncQueue {
  KeyValueDB::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  KVGetAsyncQueueImpl(KeyValueDB::Stub &Stub) : Stub(Stub) {}

  void getValueAsyncImpl(std::string Key,
                         std::shared_ptr<AsyncCallerContext> CallCtx) override {
    GetValueRequest Request;
    Request.set_key(std::move(Key));

    auto *call = new AsyncClientCall<GetValueResponse>();
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncGetValue(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<AsyncClientCall<GetValueResponse> *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);
    if (call->Response.has_error())
      return createStringError(inconvertibleErrorCode(),
                               call->Response.error().description());
    if (call->Response.has_value()) {
      Resp.Value.emplace();
      KeyValueDBClient::ValueTy &ClientValue = *Resp.Value;
      const Value &V = call->Response.value();
      for (const auto &Entry : V.entries()) {
        ClientValue[Entry.first] = Entry.second;
      }
    }
    return Resp;
  }
};

class KVPutAsyncQueueImpl : public KeyValueDBClient::PutValueAsyncQueue {
  KeyValueDB::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  KVPutAsyncQueueImpl(KeyValueDB::Stub &Stub) : Stub(Stub) {}

  void putValueAsyncImpl(std::string Key,
                         const KeyValueDBClient::ValueTy &Value,
                         std::shared_ptr<AsyncCallerContext> CallCtx) override {
    PutValueRequest Request;
    Request.set_key(std::move(Key));
    auto &PBMap = *Request.mutable_value()->mutable_entries();
    for (const auto &Entry : Value) {
      PBMap[Entry.first().str()] = Entry.second;
    }

    auto *call = new AsyncClientCall<PutValueResponse>();
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncPutValue(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<AsyncClientCall<PutValueResponse> *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);
    if (call->Response.has_error())
      return createStringError(inconvertibleErrorCode(),
                               call->Response.error().description());
    return Resp;
  }
};

struct CASLoadBlobClientCall : public AsyncClientCall<CASLoadResponse> {
  Optional<std::string> OutFilePath;
};

class CASLoadAsyncQueueImpl : public CASDBClient::LoadAsyncQueue {
  CASDBService::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  CASLoadAsyncQueueImpl(CASDBService::Stub &Stub) : Stub(Stub) {}

  void loadAsyncImpl(std::string CASID, Optional<std::string> OutFilePath,
                     std::shared_ptr<AsyncCallerContext> CallCtx) override {
    assert(!OutFilePath || !OutFilePath->empty());
    CASLoadRequest Request;
    Request.mutable_cas_id()->set_id(std::move(CASID));
    Request.set_write_to_disk(OutFilePath.has_value());

    auto *call = new CASLoadBlobClientCall();
    call->OutFilePath = std::move(OutFilePath);
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncLoad(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<CASLoadBlobClientCall *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);
    switch (call->Response.outcome()) {
    case CASLoadResponse_Outcome_OBJECT_NOT_FOUND:
      Resp.KeyNotFound = true;
      return Resp;
    case CASLoadResponse_Outcome_ERROR:
      return createStringError(inconvertibleErrorCode(),
                               call->Response.error().description());
    default:
      break;
    }
    assert(call->Response.outcome() == CASLoadResponse_Outcome_SUCCESS);
    const auto &Blob = call->Response.data().blob();
    if (call->OutFilePath.has_value()) {
      if (Blob.has_file_path()) {
        if (std::error_code EC =
                sys::fs::rename(Blob.file_path(), *call->OutFilePath))
          return createStringError(EC, "failed rename '" + Blob.file_path() +
                                           "' to '" + *call->OutFilePath +
                                           "': " + EC.message());
      } else {
        std::error_code EC;
        raw_fd_ostream OS(*call->OutFilePath, EC);
        if (EC)
          return createStringError(EC, "failed creating ''" +
                                           *call->OutFilePath +
                                           "': " + EC.message());
        OS << Blob.data();
        OS.close();
      }
    } else {
      if (Blob.has_file_path()) {
        ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuf =
            MemoryBuffer::getFile(Blob.file_path());
        if (!FileBuf)
          return createStringError(FileBuf.getError(),
                                   "failed reading '" + Blob.file_path() +
                                       "': " + FileBuf.getError().message());
        Resp.BlobData = (*FileBuf)->getBuffer().str();
      } else {
        Resp.BlobData = Blob.data();
      }
    }
    return Resp;
  }
};

class CASSaveAsyncQueueImpl : public CASDBClient::SaveAsyncQueue {
  CASDBService::Stub &Stub;
  grpc::CompletionQueue CQ;

public:
  CASSaveAsyncQueueImpl(CASDBService::Stub &Stub) : Stub(Stub) {}

  void saveDataAsyncImpl(std::string BlobData,
                         std::shared_ptr<AsyncCallerContext> CallCtx) override {
    CASSaveRequest Request;
    Request.mutable_data()->mutable_blob()->set_data(std::move(BlobData));
    casSaveAsync(Request, std::move(CallCtx));
  }

  void saveFileAsyncImpl(std::string FilePath,
                         std::shared_ptr<AsyncCallerContext> CallCtx) override {
    assert(!FilePath.empty());
    CASSaveRequest Request;
    Request.mutable_data()->mutable_blob()->set_file_path(std::move(FilePath));
    casSaveAsync(Request, std::move(CallCtx));
  }

  void casSaveAsync(const CASSaveRequest &Request,
                    std::shared_ptr<AsyncCallerContext> CallCtx) {
    auto *call = new AsyncClientCall<CASSaveResponse>();
    call->CallCtx = std::move(CallCtx);
    call->Reader = Stub.AsyncSave(&call->Context, Request, &CQ);
    call->Reader->Finish(&call->Response, &call->Status, call);
  }

  Expected<Response> receiveNextImpl() override {
    void *got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue.
    CQ.Next(&got_tag, &ok);
    // The tag in this example is the memory location of the call object
    auto *call = static_cast<AsyncClientCall<CASSaveResponse> *>(got_tag);
    auto _ = llvm::make_scope_exit([&]() { delete call; });

    if (!call->Status.ok())
      return errorFromGRPCStatus(call->Status);
    if (!ok)
      return createStringError(inconvertibleErrorCode(),
                               "service channel shutdown");

    Response Resp;
    Resp.CallCtx = std::move(call->CallCtx);
    if (call->Response.has_error())
      return createStringError(inconvertibleErrorCode(),
                               call->Response.error().description());
    Resp.CASID = call->Response.cas_id().id();
    return Resp;
  }
};

class KeyValueDBClientImpl : public KeyValueDBClient {
  std::unique_ptr<KeyValueDB::Stub> Stub;

public:
  KeyValueDBClientImpl(std::shared_ptr<grpc::Channel> Channel)
      : Stub(KeyValueDB::NewStub(std::move(Channel))) {
    GetValueQueue = std::make_unique<KVGetAsyncQueueImpl>(*Stub);
    PutValueQueue = std::make_unique<KVPutAsyncQueueImpl>(*Stub);
  }
};

class CASDBClientImpl : public CASDBClient {
  std::unique_ptr<CASDBService::Stub> Stub;

public:
  CASDBClientImpl(std::shared_ptr<grpc::Channel> Channel)
      : Stub(CASDBService::NewStub(std::move(Channel))) {
    LoadQueue = std::make_unique<CASLoadAsyncQueueImpl>(*Stub);
    SaveQueue = std::make_unique<CASSaveAsyncQueueImpl>(*Stub);
  }
};

} // namespace

Expected<ClientServices>
remote_cache::createCompilationCachingRemoteClient(StringRef SocketPath) {
  std::string Address("unix:");
  Address += SocketPath;
  const auto Channel = grpc::CreateChannel(std::move(Address),
                                           grpc::InsecureChannelCredentials());
  auto KVClient = std::make_unique<KeyValueDBClientImpl>(Channel);
  auto CASClient = std::make_unique<CASDBClientImpl>(std::move(Channel));
  return ClientServices{std::move(KVClient), std::move(CASClient)};
}
