//===--- GenerateSetter.cpp - Generate setter methods ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Config.h"
#include "refactor/GenerateAccessorBase.cpp"
#include "refactor/GenerateAccessorBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/TypeBase.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include <string>

static constexpr const char *SetterPrefixDefault = "set";
static constexpr const char *SetterParameterPrefixDefault = "new";

namespace clang {
namespace clangd {
namespace {

// A tweak that generates a setter for a field.
//
// Given:
//   struct S { int x; };
// Produces:
//   void setX(int newX) { x = newX; }
//
// Method's prefix can be configured with Style.SetterPrefix.
// Method's parameter prefix can be configured with
// Style.SetterParameterPrefix.
//
// We place the method inline, other tweaks are available to outline it.
class GenerateSetter : public GenerateAccessorBase {
public:
  const char *id() const final;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }
  std::string title() const override { return "Generate setter"; }

private:
  std::string buildCode() const override {
    QualType T = Field->getType().getLocalUnqualifiedType();
    auto &Context = Class->getASTContext();

    std::string S;
    llvm::raw_string_ostream OS(S);

    OS << "void " << AccessorName << "(";

    // Use const-ref if type is not trivially copiable or its size is larger
    // than the size of a pointer
    if (!T.isTriviallyCopyableType(Context) ||
        Context.getTypeSize(T) > Context.getTypeSize(Context.VoidPtrTy)) {
      OS << "const " << printType(T, *Class);
      if (!T->isReferenceType())
        OS << " &";
    } else
      OS << printType(T, *Class) << " ";

    std::string SetterParameterPrefix =
        Config::current().Style.SetterParameterPrefix;
    if (SetterParameterPrefix.empty() && !FieldWithPreffixOrSuffix) {
      SetterParameterPrefix = SetterParameterPrefixDefault;
    }
    llvm::StringRef FieldName = Field->getName();
    OS << SetterParameterPrefix << FieldBaseName << ") { " << FieldName << " = "
       << SetterParameterPrefix << FieldBaseName << "; }\n";
    return S;
  }

  std::string retrieveAccessorPrefix() const override {
    // Get user-configured setter prefix
    std::string SetterPrefix = Config::current().Style.SetterPrefix;
    if (SetterPrefix.empty()) {
      return SetterPrefixDefault;
    }
    return SetterPrefix;
  }

}; // namespace

REGISTER_TWEAK(GenerateSetter)

} // namespace
} // namespace clangd
} // namespace clang
