#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_GENERATEACCESSORBASE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_GENERATEACCESSORBASE_H

//===--- GenerateAccessorBase.h ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/Tweak.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang {
namespace clangd {

class GenerateAccessorBase : public Tweak {
public:
  const char *id() const override = 0;
  llvm::StringLiteral kind() const override = 0;
  std::string title() const override = 0;

  bool prepare(const Selection &Inputs) override;

  Expected<Effect> apply(const Selection &Inputs) override;

protected:
  virtual std::string buildCode() const = 0;

  void build();

  virtual std::string retrieveAccessorPrefix() const = 0;

  void toLower(std::string &s) const;

  void toUpper(std::string &s) const;

  void toCamelCase(std::string &s) const;

  const CXXRecordDecl *Class = nullptr;
  const FieldDecl *Field = nullptr;
  std::string AccessorName;
  std::string FieldBaseName;
  bool FieldWithPreffixOrSuffix = false;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_GENERATEACCESSORBASE_H
