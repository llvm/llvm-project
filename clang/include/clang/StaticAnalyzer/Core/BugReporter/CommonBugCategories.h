//=--- CommonBugCategories.h - Provides common issue categories -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_COMMONBUGCATEGORIES_H
#define LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_COMMONBUGCATEGORIES_H

// Common strings used for the "category" of many static analyzer issues.
#include "clang/Support/Compiler.h"
namespace clang {
namespace ento {
namespace categories {
CLANG_ABI extern const char *const AppleAPIMisuse;
CLANG_ABI extern const char *const CoreFoundationObjectiveC;
CLANG_ABI extern const char *const LogicError;
CLANG_ABI extern const char *const MemoryRefCount;
CLANG_ABI extern const char *const MemoryError;
CLANG_ABI extern const char *const UnixAPI;
CLANG_ABI extern const char *const CXXObjectLifecycle;
CLANG_ABI extern const char *const CXXMoveSemantics;
CLANG_ABI extern const char *const SecurityError;
CLANG_ABI extern const char *const UnusedCode;
CLANG_ABI extern const char *const TaintedData;
} // namespace categories
} // namespace ento
} // namespace clang
#endif
