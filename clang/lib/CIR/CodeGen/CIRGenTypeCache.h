//===--- CIRGenTypeCache.h - Commonly used LLVM types and info -*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This structure provides a set of common types useful during CIR emission.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENTYPECACHE_H
#define LLVM_CLANG_LIB_CIR_CIRGENTYPECACHE_H

namespace clang::CIRGen {

/// This structure provides a set of types that are commonly used
/// during IR emission. It's initialized once in CodeGenModule's
/// constructor and then copied around into new CIRGenFunction's.
struct CIRGenTypeCache {
  CIRGenTypeCache() = default;
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENTYPECACHE_H
