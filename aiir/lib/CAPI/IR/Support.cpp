//===- Support.cpp - Helpers for C interface to AIIR API ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/CAPI/Support.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <string>

AiirStringRef aiirStringRefCreateFromCString(const char *str) {
  return aiirStringRefCreate(str, strlen(str));
}

bool aiirStringRefEqual(AiirStringRef string, AiirStringRef other) {
  return llvm::StringRef(string.data, string.length) ==
         llvm::StringRef(other.data, other.length);
}

//===----------------------------------------------------------------------===//
// LLVM ThreadPool API.
//===----------------------------------------------------------------------===//
AiirLlvmThreadPool aiirLlvmThreadPoolCreate() {
  return wrap(new llvm::DefaultThreadPool());
}

void aiirLlvmThreadPoolDestroy(AiirLlvmThreadPool threadPool) {
  delete unwrap(threadPool);
}

int aiirLlvmThreadPoolGetMaxConcurrency(AiirLlvmThreadPool threadPool) {
  return unwrap(threadPool)->getMaxConcurrency();
}

//===----------------------------------------------------------------------===//
// LLVM raw_fd_ostream API.
//===----------------------------------------------------------------------===//

AiirLlvmRawFdOStream
aiirLlvmRawFdOStreamCreate(const char *path, bool binary,
                           AiirStringCallback errorCallback, void *userData) {
  std::error_code ec;
  auto flags = binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_Text;
  auto *stream = new llvm::raw_fd_ostream(path, ec, flags);
  if (ec) {
    delete stream;
    if (errorCallback) {
      std::string message = ec.message();
      errorCallback(aiirStringRefCreate(message.data(), message.size()),
                    userData);
    }
    return wrap(static_cast<llvm::raw_fd_ostream *>(nullptr));
  }
  return wrap(stream);
}

void aiirLlvmRawFdOStreamWrite(AiirLlvmRawFdOStream stream,
                               AiirStringRef string) {
  unwrap(stream)->write(string.data, string.length);
}

bool aiirLlvmRawFdOStreamIsNull(AiirLlvmRawFdOStream stream) {
  return !stream.ptr;
}

void aiirLlvmRawFdOStreamDestroy(AiirLlvmRawFdOStream stream) {
  delete unwrap(stream);
}

//===----------------------------------------------------------------------===//
// TypeID API.
//===----------------------------------------------------------------------===//
AiirTypeID aiirTypeIDCreate(const void *ptr) {
  assert(reinterpret_cast<uintptr_t>(ptr) % 8 == 0 &&
         "ptr must be 8 byte aligned");
  // This is essentially a no-op that returns back `ptr`, but by going through
  // the `TypeID` functions we can get compiler errors in case the `TypeID`
  // api/representation changes
  return wrap(aiir::TypeID::getFromOpaquePointer(ptr));
}

bool aiirTypeIDEqual(AiirTypeID typeID1, AiirTypeID typeID2) {
  return unwrap(typeID1) == unwrap(typeID2);
}

size_t aiirTypeIDHashValue(AiirTypeID typeID) {
  return hash_value(unwrap(typeID));
}

//===----------------------------------------------------------------------===//
// TypeIDAllocator API.
//===----------------------------------------------------------------------===//

AiirTypeIDAllocator aiirTypeIDAllocatorCreate() {
  return wrap(new aiir::TypeIDAllocator());
}

void aiirTypeIDAllocatorDestroy(AiirTypeIDAllocator allocator) {
  delete unwrap(allocator);
}

AiirTypeID aiirTypeIDAllocatorAllocateTypeID(AiirTypeIDAllocator allocator) {
  return wrap(unwrap(allocator)->allocate());
}
