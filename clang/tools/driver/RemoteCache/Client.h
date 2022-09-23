//===-- RemoteCache/Client.h - Remote Client --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// gRPC client for the remote cache service protocol. It provides asynchronous
// APIs to allow initiating multiple concurrent requests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_DRIVER_REMOTECACHE_CLIENT_H
#define LLVM_CLANG_TOOLS_DRIVER_REMOTECACHE_CLIENT_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace remote_cache {

/// Used to optionally associate additional context with a particular request.
class AsyncCallerContext {
  virtual void anchor();

public:
  virtual ~AsyncCallerContext() = default;
};

class AsyncQueueBase {
  virtual void anchor();

public:
  virtual ~AsyncQueueBase() = default;

  bool hasPending() const { return NumPending != 0; }

protected:
  unsigned NumPending = 0;
};

/// An asynchronous gRPC client for the key-value service of the remote cache
/// protocol.
///
/// Example usage:
/// \code
///   // Initiate one (or more) `GetValue` request.
///   KVClient->getValueQueue().getValueAsync(ResultCacheKey.getHash());
///   // Wait for one response.
///   auto Response = KVClient->getValueQueue().receiveNext();
/// \endcode
class KeyValueDBClient {
  virtual void anchor();

public:
  virtual ~KeyValueDBClient() = default;

  using ValueTy = llvm::StringMap<std::string>;

  class GetValueAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~GetValueAsyncQueue() = default;

    void getValueAsync(std::string Key,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      getValueAsyncImpl(std::move(Key), std::move(CallCtx));
      ++NumPending;
    }
    void getValueAsync(ArrayRef<uint8_t> Key,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      getValueAsync(toStringRef(Key).str(), std::move(CallCtx));
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
      // If this is \p None it means the key was not found.
      Optional<ValueTy> Value;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void
    getValueAsyncImpl(std::string Key,
                      std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  class PutValueAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~PutValueAsyncQueue() = default;

    void putValueAsync(std::string Key, const ValueTy &Value,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      putValueAsyncImpl(std::move(Key), Value, std::move(CallCtx));
      ++NumPending;
    }
    void putValueAsync(ArrayRef<uint8_t> Key, const ValueTy &Value,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      putValueAsync(toStringRef(Key).str(), Value, std::move(CallCtx));
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void
    putValueAsyncImpl(std::string Key, const ValueTy &Value,
                      std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  GetValueAsyncQueue &getValueQueue() const { return *GetValueQueue; }
  PutValueAsyncQueue &putValueQueue() const { return *PutValueQueue; }

protected:
  std::unique_ptr<GetValueAsyncQueue> GetValueQueue;
  std::unique_ptr<PutValueAsyncQueue> PutValueQueue;
};

/// An asynchronous gRPC client for the CAS service of the remote cache
/// protocol.
///
/// FIXME: This only implements the \p CASSave and \p CASLoad APIs.
///
/// Example usage:
/// \code
///   auto &LoadQueue = CASClient->loadQueue();
///   // Initiate one or more `CASLoad` requests.
///   LoadQueue.loadAsync(CASID, Path, std::make_shared<CallCtx>(OutputName));
///   // Wait for responses.
///   while (LoadQueue.hasPending()) {
///     auto Response = LoadQueue.receiveNext();
///     ...
///   }
/// \endcode
class CASDBClient {
  virtual void anchor();

public:
  virtual ~CASDBClient() = default;

  class LoadAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~LoadAsyncQueue() = default;

    void loadAsync(std::string CASID, Optional<std::string> OutFilePath = None,
                   std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      loadAsyncImpl(std::move(CASID), std::move(OutFilePath),
                    std::move(CallCtx));
      ++NumPending;
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
      bool KeyNotFound = false;
      Optional<std::string> BlobData;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void loadAsyncImpl(std::string CASID,
                               Optional<std::string> OutFilePath,
                               std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  class SaveAsyncQueue : public AsyncQueueBase {
    virtual void anchor() override;

  public:
    virtual ~SaveAsyncQueue() = default;

    void saveDataAsync(std::string BlobData,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      saveDataAsyncImpl(std::move(BlobData), std::move(CallCtx));
      ++NumPending;
    }

    void saveFileAsync(std::string FilePath,
                       std::shared_ptr<AsyncCallerContext> CallCtx = nullptr) {
      saveFileAsyncImpl(std::move(FilePath), std::move(CallCtx));
      ++NumPending;
    }

    struct Response {
      std::shared_ptr<AsyncCallerContext> CallCtx;
      std::string CASID;
    };
    Expected<Response> receiveNext() {
      assert(NumPending);
      --NumPending;
      return receiveNextImpl();
    }

  protected:
    virtual void
    saveDataAsyncImpl(std::string BlobData,
                      std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual void
    saveFileAsyncImpl(std::string FilePath,
                      std::shared_ptr<AsyncCallerContext> CallCtx) = 0;
    virtual Expected<Response> receiveNextImpl() = 0;
  };

  LoadAsyncQueue &loadQueue() const { return *LoadQueue; }
  SaveAsyncQueue &saveQueue() const { return *SaveQueue; }

protected:
  std::unique_ptr<LoadAsyncQueue> LoadQueue;
  std::unique_ptr<SaveAsyncQueue> SaveQueue;
};

struct ClientServices {
  std::unique_ptr<KeyValueDBClient> KVDB;
  std::unique_ptr<CASDBClient> CASDB;
};
Expected<ClientServices>
createCompilationCachingRemoteClient(StringRef SocketPath);

} // namespace remote_cache
} // namespace clang

#endif
