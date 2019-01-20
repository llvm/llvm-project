//===- TaintTag.h - Path-sensitive "State" for tracking values --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a set of taint tags. Several tags are used to differentiate kinds
// of taint.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_TAINTTAG_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_TAINTTAG_H

namespace clang {
namespace ento {

/// The type of taint, which helps to differentiate between different types of
/// taint.
using TaintTagType = unsigned;

static const TaintTagType TaintTagGeneric = 0;

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_TAINTTAG_H
