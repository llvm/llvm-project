//===--- Attributes.h - Attributes header -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ATTRIBUTES_H
#define LLVM_CLANG_BASIC_ATTRIBUTES_H

#include "clang/Basic/AttributeCommonInfo.h"

namespace clang {

class IdentifierInfo;
class LangOptions;
class TargetInfo;

/// Return the version number associated with the attribute if we
/// recognize and implement the attribute specified by the given information.
int hasAttribute(AttributeCommonInfo::Syntax Syntax,
                 const IdentifierInfo *Scope, const IdentifierInfo *Attr,
                 const TargetInfo &Target, const LangOptions &LangOpts);

inline const char* deuglifyAttrScope(StringRef Scope) {
  if (Scope == "_Clang")
    return "clang";
  if (Scope == "__gnu__")
    return "gnu";
  if (Scope == "__msvc__")
    return "msvc";
  return nullptr;
}

inline const char* uglifyAttrScope(StringRef Scope) {
  if (Scope == "clang")
    return "_Clang";
  if (Scope == "gnu")
    return "__gnu__";
  if (Scope == "msvc")
    return "__msvc__";
  return nullptr;
}

inline bool isPotentiallyUglyScope(StringRef Scope) {
  return Scope == "gnu" || Scope == "clang" || Scope == "msvc";
}

} // end namespace clang

#endif // LLVM_CLANG_BASIC_ATTRIBUTES_H
