//===- Support.h - C API Helpers Implementation -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for converting AIIR C++ objects into helper
// C structures for the purpose of C API. This file should not be included from
// C++ code other than C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CAPI_SUPPORT_H
#define AIIR_CAPI_SUPPORT_H

#include "aiir-c/Support.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/Support/TypeID.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace llvm {
class ThreadPoolInterface;
class raw_fd_ostream;
} // namespace llvm

/// Converts a StringRef into its AIIR C API equivalent.
inline AiirStringRef wrap(llvm::StringRef ref) {
  return aiirStringRefCreate(ref.data(), ref.size());
}

/// Creates a StringRef out of its AIIR C API equivalent.
inline llvm::StringRef unwrap(AiirStringRef ref) {
  return llvm::StringRef(ref.data, ref.length);
}

inline AiirLogicalResult wrap(llvm::LogicalResult res) {
  if (aiir::succeeded(res))
    return aiirLogicalResultSuccess();
  return aiirLogicalResultFailure();
}

inline llvm::LogicalResult unwrap(AiirLogicalResult res) {
  return aiir::success(aiirLogicalResultIsSuccess(res));
}

DEFINE_C_API_PTR_METHODS(AiirLlvmThreadPool, llvm::ThreadPoolInterface)
DEFINE_C_API_PTR_METHODS(AiirLlvmRawFdOStream, llvm::raw_fd_ostream)
DEFINE_C_API_METHODS(AiirTypeID, aiir::TypeID)
DEFINE_C_API_PTR_METHODS(AiirTypeIDAllocator, aiir::TypeIDAllocator)

#endif // AIIR_CAPI_SUPPORT_H
