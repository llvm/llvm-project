//===- FaultingCAS.h - Test helper that injects CAS failures ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_UNITTESTS_FRONTEND_FAULTINGCAS_H
#define CLANG_UNITTESTS_FRONTEND_FAULTINGCAS_H

#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {

/// Test wrapper that delegates to an inner CAS but can inject an error on the
/// N-th call to \c store(). All other mutation/query operations are passed
/// through unchanged. The handle-based read hooks are intentionally not
/// implemented: tests that drive this wrapper short-circuit on the injected
/// store failure before any object is read.
class FaultingCAS : public llvm::cas::ObjectStore {
public:
  FaultingCAS(std::unique_ptr<llvm::cas::ObjectStore> Inner,
              unsigned FailStoreAtCall)
      : ObjectStore(Inner->getContext()), Inner(std::move(Inner)),
        FailStoreAtCall(FailStoreAtCall) {}

  unsigned getStoreCallCount() const { return StoreCallCount; }

  llvm::Expected<llvm::cas::CASID> parseID(llvm::StringRef ID) override {
    return Inner->parseID(ID);
  }
  llvm::Expected<llvm::cas::ObjectRef>
  store(llvm::ArrayRef<llvm::cas::ObjectRef> Refs,
        llvm::ArrayRef<char> Data) override {
    if (StoreCallCount++ == FailStoreAtCall)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "injected store error");
    return Inner->store(Refs, Data);
  }
  llvm::cas::CASID getID(llvm::cas::ObjectRef Ref) const override {
    return Inner->getID(Ref);
  }
  std::optional<llvm::cas::ObjectRef>
  getReference(const llvm::cas::CASID &ID) const override {
    return Inner->getReference(ID);
  }
  llvm::Expected<bool> isMaterialized(llvm::cas::ObjectRef Ref) const override {
    return Inner->isMaterialized(Ref);
  }
  llvm::Error validateObject(const llvm::cas::CASID &ID) override {
    return Inner->validateObject(ID);
  }
  llvm::Error validate(bool CheckHash) const override {
    return Inner->validate(CheckHash);
  }

protected:
  llvm::Expected<std::optional<llvm::cas::ObjectHandle>>
  loadIfExists(llvm::cas::ObjectRef) override {
    llvm::report_fatal_error("FaultingCAS: loadIfExists not implemented");
  }
  uint64_t getDataSize(llvm::cas::ObjectHandle) const override {
    llvm::report_fatal_error("FaultingCAS: getDataSize not implemented");
  }
  llvm::Error forEachRef(
      llvm::cas::ObjectHandle,
      llvm::function_ref<llvm::Error(llvm::cas::ObjectRef)>) const override {
    llvm::report_fatal_error("FaultingCAS: forEachRef not implemented");
  }
  llvm::cas::ObjectRef readRef(llvm::cas::ObjectHandle,
                               size_t) const override {
    llvm::report_fatal_error("FaultingCAS: readRef not implemented");
  }
  size_t getNumRefs(llvm::cas::ObjectHandle) const override {
    llvm::report_fatal_error("FaultingCAS: getNumRefs not implemented");
  }
  llvm::ArrayRef<char> getData(llvm::cas::ObjectHandle, bool) const override {
    llvm::report_fatal_error("FaultingCAS: getData not implemented");
  }

private:
  std::unique_ptr<llvm::cas::ObjectStore> Inner;
  unsigned StoreCallCount = 0;
  unsigned FailStoreAtCall;
};

} // namespace clang

#endif
