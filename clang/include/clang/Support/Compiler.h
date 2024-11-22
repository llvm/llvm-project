//===-- clang/Support/Compiler.h - Compiler abstraction support -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines explicit visibility macros used to export symbols from
// clang-cpp
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_SUPPORT_COMPILER_H
#define CLANG_SUPPORT_COMPILER_H

#include "llvm/Support/Compiler.h"

/// CLANG_ABI is the main export/visibility macro to mark something as
/// explicitly exported when clang is built as a shared library with everything
/// else that is unannotated having hidden visibility.
///
/// CLANG_EXPORT_TEMPLATE is used on explicit template instantiations in source
/// files that were declared extern in a header. This macro is only set as a
/// compiler export attribute on windows, on other platforms it does nothing.
///
/// CLANG_TEMPLATE_ABI is for annotating extern template declarations in headers
/// for both functions and classes. On windows its turned in to dllimport for
/// library consumers, for other platforms its a default visibility attribute.
#ifndef CLANG_ABI_GENERATING_ANNOTATIONS
// Marker to add to classes or functions in public headers that should not have
// export macros added to them by the clang tool
#define CLANG_ABI_NOT_EXPORTED
// Some libraries like those for tablegen are linked in to tools that used
// in the build so can't depend on the llvm shared library. If export macros
// were left enabled when building these we would get duplicate or
// missing symbol linker errors on windows.
#if defined(CLANG_BUILD_STATIC)
#define CLANG_ABI
#define CLANG_TEMPLATE_ABI
#define CLANG_EXPORT_TEMPLATE
#elif defined(_WIN32) && !defined(__MINGW32__)
#if defined(CLANG_EXPORTS)
#define CLANG_ABI __declspec(dllexport)
#define CLANG_TEMPLATE_ABI
#define CLANG_EXPORT_TEMPLATE __declspec(dllexport)
#else
#define CLANG_ABI __declspec(dllimport)
#define CLANG_TEMPLATE_ABI __declspec(dllimport)
#define CLANG_EXPORT_TEMPLATE
#endif
#elif defined(__ELF__) || defined(__MINGW32__) || defined(_AIX) ||             \
    defined(__MVS__)
#define CLANG_ABI LLVM_ATTRIBUTE_VISIBILITY_DEFAULT
#define CLANG_TEMPLATE_ABI LLVM_ATTRIBUTE_VISIBILITY_DEFAULT
#define CLANG_EXPORT_TEMPLATE
#elif defined(__MACH__) || defined(__WASM__)
#define CLANG_ABI LLVM_ATTRIBUTE_VISIBILITY_DEFAULT
#define CLANG_TEMPLATE_ABI
#define CLANG_EXPORT_TEMPLATE
#endif
#endif

#endif
