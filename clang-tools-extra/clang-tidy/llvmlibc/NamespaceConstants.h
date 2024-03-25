//===--- NamespaceConstants.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"

namespace clang::tidy::llvm_libc {

const static llvm::StringRef RequiredNamespaceStart = "__llvm_libc";
const static llvm::StringRef RequiredNamespaceMacroName = "LIBC_NAMESPACE";

} // namespace clang::tidy::llvm_libc
