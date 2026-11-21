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

#include "clang/Config/clang-config.h"

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
///
/// CLANG_ABI_EXPORT always exports (or, in a static build, does nothing), it
/// never expands to dllimport. It is the clang analogue of LLVM_ABI_EXPORT and
/// is used where a symbol must be exported by the defining library but must not
/// be imported by consumers.
#ifndef CLANG_ABI_GENERATING_ANNOTATIONS

// Marker to add to classes or functions in public headers that should not have
// export macros added to them by the clang tool
#define CLANG_ABI_NOT_EXPORTED

// clang-format off
#if defined(CLANG_ENABLE_CLANG_EXPORT_ANNOTATIONS)
# if defined(_WIN32) && !defined(__MINGW32__)
#  if defined(CLANG_EXPORTS)
#   define CLANG_ABI __declspec(dllexport)
#   define CLANG_TEMPLATE_ABI
#   define CLANG_EXPORT_TEMPLATE __declspec(dllexport)
#  else
#   define CLANG_ABI __declspec(dllimport)
#   define CLANG_TEMPLATE_ABI __declspec(dllimport)
#   define CLANG_EXPORT_TEMPLATE
#  endif
#  define CLANG_ABI_EXPORT __declspec(dllexport)
# else
#  define CLANG_ABI __attribute__((visibility("default")))
#  define CLANG_TEMPLATE_ABI CLANG_ABI
#  define CLANG_EXPORT_TEMPLATE
#  define CLANG_ABI_EXPORT CLANG_ABI
# endif
#else
# define CLANG_ABI
# define CLANG_TEMPLATE_ABI
# define CLANG_EXPORT_TEMPLATE
# define CLANG_ABI_EXPORT
#endif
// clang-format on

#endif

#endif
