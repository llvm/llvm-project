//===--- GenerateGetter.cpp - Generate getter methods ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Config.h"
#include "refactor/GenerateAccessorBase.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include <string>

static constexpr const char *GetterPrefixDefault = "get";

namespace clang {
namespace clangd {
namespace {

// A tweak that generates a getter for a field.
//
// Given:
//   struct S { int x; };
// Produces:
//   int getX() const { return x; }
//
// Method's prefix can be configured with Style.GetterPrefix.
//
// We place the method inline, other tweaks are available to outline it.
class GenerateGetter : public GenerateAccessorBase {
public:
  const char *id() const final;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }
  std::string title() const override { return "Generate getter"; }

private:
  std::string buildCode() const override {
    // Field type
    QualType T = Field->getType().getLocalUnqualifiedType();

    std::string S;
    llvm::raw_string_ostream OS(S);
    llvm::StringRef FieldName = Field->getName();

    OS << printType(T, *Class) << " " << AccessorName << "() const { return "
       << FieldName << "; }\n";
    return S;
  }

  std::string retrieveAccessorPrefix() const override {
    // Get user-configured getter prefix.
    std::string GetterPrefix = Config::current().Style.GetterPrefix;
    if (!GetterPrefix.empty()) {
      return GetterPrefix;
    }
    if (FieldWithPreffixOrSuffix) {
      return "";
    }
    return GetterPrefixDefault;
  }
};

REGISTER_TWEAK(GenerateGetter)

} // namespace
} // namespace clangd
} // namespace clang
