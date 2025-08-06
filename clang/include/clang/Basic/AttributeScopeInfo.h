//==- AttributeScopeInfo.h - Base info about an Attribute Scope --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeScopeInfo type, which represents information
// about the scope of an attribute.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ATTRIBUTESCOPEINFO_H
#define LLVM_CLANG_BASIC_ATTRIBUTESCOPEINFO_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

class IdentifierInfo;

class AttributeScopeInfo {
public:
  AttributeScopeInfo() = default;

  AttributeScopeInfo(const IdentifierInfo *Name, SourceLocation NameLoc)
      : Name(Name), NameLoc(NameLoc) {}

  AttributeScopeInfo(const IdentifierInfo *Name, SourceLocation NameLoc,
                     SourceLocation CommonScopeLoc)
      : Name(Name), NameLoc(NameLoc), CommonScopeLoc(CommonScopeLoc) {}

  const IdentifierInfo *getName() const { return Name; }
  SourceLocation getNameLoc() const { return NameLoc; }

  bool isValid() const { return Name != nullptr; }
  bool isExplicit() const { return CommonScopeLoc.isInvalid(); }

private:
  const IdentifierInfo *Name = nullptr;
  SourceLocation NameLoc;
  SourceLocation CommonScopeLoc;
};

} // namespace clang

#endif // LLVM_CLANG_BASIC_ATTRIBUTESCOPEINFO_H
