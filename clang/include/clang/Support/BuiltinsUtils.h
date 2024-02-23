//===--- BuiltinsUtils.h - clang Builtins Utils -*- C++ -*-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_SUPPORT_BUILTINSUTILS_H
#define CLANG_SUPPORT_BUILTINSUTILS_H

#include "llvm/ADT/StringRef.h"
#include <string>
namespace llvm {
class SMLoc;
}
namespace clang {

/// Parse builtins prototypes according to the rules in
/// clang/include/clang/Basic/Builtins.def
void ParseBuiltinType(llvm::StringRef T, llvm::StringRef Substitution,
                      std::string &Type, llvm::SMLoc *Loc);

} // namespace clang
#endif // CLANG_SUPPORT_BUILTINSUTILS_H
