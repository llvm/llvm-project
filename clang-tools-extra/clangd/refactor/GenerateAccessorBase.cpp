//===--- GenerateAccessorBase.cpp --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GenerateAccessorBase.h"
#include "AST.h"
#include "Config.h"
#include "ParsedAST.h"
#include "refactor/InsertionPoint.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>

namespace clang {
namespace clangd {

Expected<Tweak::Effect> GenerateAccessorBase::apply(const Selection &Inputs) {
  // Prefer to place the new method ...
  std::vector<Anchor> Anchors = {
      // On top of fields declaration
      {[](const Decl *D) { return llvm::isa<FieldDecl>(D); }, Anchor::Above},
      // At the bottom of public section
      {[](const Decl *D) { return D->getAccess() == AS_public; },
       Anchor::Below},
      // Fallback: At the end of class
      {[](const Decl *D) { return true; }, Anchor::Below},
  };

  std::string Code = buildCode();

  auto Edit = insertDecl(Code, *Class, std::move(Anchors), AS_public);
  if (!Edit)
    return Edit.takeError();

  return Effect::mainFileEdit(Inputs.AST->getSourceManager(),
                              tooling::Replacements{std::move(*Edit)});
}

bool GenerateAccessorBase::prepare(const Selection &Inputs) {
  // This tweak is available for C++ only.
  if (!Inputs.AST->getLangOpts().CPlusPlus)
    return false;

  if (auto *N = Inputs.ASTSelection.commonAncestor()) {
    Field = N->ASTNode.get<FieldDecl>();
  }

  // Trigger only on Field declaration.
  if (!Field || !Field->getIdentifier())
    return false;

  // No setter for constant field, by extension no action for generate getter
  // will be provided.
  if (Field->getType().isConstQualified())
    return false;

  // Trigger only inside a class declaration.
  Class = dyn_cast<CXXRecordDecl>(Field->getParent());
  if (!Class || !Class->isThisDeclarationADefinition())
    return false;

  // Define accessor's name, field's base name and check if field has any
  // prefix or suffix.
  build();

  // Trigger only if the class does not already have this method.
  for (const auto *M : Class->methods()) {
    if (M->getName() == AccessorName) {
      return false;
    }
  }

  dlog("GenerateAccessorBase for {0}?", Field->getName());

  return true;
}

void GenerateAccessorBase::build() {
  // Retrieve clang-tidy options for field's prefix and suffix in order to
  // determine the base name of the field. Do not take Hungarian notation into
  // account.
  const auto &ClangTiddyOptions =
      Config::current().Diagnostics.ClangTidy.CheckOptions;

  auto GetOption = [&ClangTiddyOptions](llvm::StringRef Key) -> std::string {
    auto It =
        ClangTiddyOptions.find("readability-identifier-naming." + Key.str());
    return It != ClangTiddyOptions.end() ? It->second : "";
  };

  std::string FieldPrefix;
  std::string FieldSuffix;
  // Visibility (public/protected/private)
  switch (Field->getAccessUnsafe()) {
  case AS_private:
    FieldPrefix = GetOption("PrivateMemberPrefix");
    FieldSuffix = GetOption("PrivateMemberSuffix");
    break;
  case AS_protected:
    FieldPrefix = GetOption("ProtectedMemberPrefix");
    FieldSuffix = GetOption("ProtectedMemberSuffix");
    break;
  case AS_public:
    FieldPrefix = GetOption("PublicMemberPrefix");
    FieldSuffix = GetOption("PublicMemberSuffix");
    break;
  case AS_none:
    break;
  }
  if (FieldPrefix.empty() && FieldSuffix.empty()) {
    FieldPrefix = GetOption("MemberPrefix");
    FieldSuffix = GetOption("MemberSuffix");
  }

  llvm::StringRef BaseNameRef = Field->getName();
  if (!FieldPrefix.empty()) {
    FieldWithPreffixOrSuffix |= BaseNameRef.consume_front(FieldPrefix);
  }
  if (!FieldSuffix.empty()) {
    FieldWithPreffixOrSuffix |= BaseNameRef.consume_back(FieldSuffix);
  }

  std::string FieldBaseNameLocal = BaseNameRef.str();
  FieldBaseName = FieldBaseNameLocal;

  // Get user-configured getter/setter prefix.
  std::string AccessorPrefix = retrieveAccessorPrefix();
  if (AccessorPrefix.empty()) {
    AccessorName = FieldBaseNameLocal;
    return;
  }

  // Define getter method case.
  std::string MethodCase = GetOption("PublicMethodCase");
  if (MethodCase.empty())
    MethodCase = GetOption("MethodCase");
  if (MethodCase.empty())
    MethodCase = GetOption("FunctionCase");

  if (MethodCase == "lower_case") {
    toLower(AccessorPrefix);
    toLower(FieldBaseNameLocal);
    AccessorName = AccessorPrefix + '_' + FieldBaseNameLocal;
    return;
  }
  if (MethodCase == "UPPER_CASE") {
    toUpper(AccessorPrefix);
    toUpper(FieldBaseNameLocal);
    AccessorName = AccessorPrefix + '_' + FieldBaseNameLocal;
    return;
  }
  if (MethodCase == "camelBack") {
    toLower(AccessorPrefix);
    toCamelCase(FieldBaseNameLocal);
    AccessorName = AccessorPrefix + FieldBaseNameLocal;
    return;
  }
  if (MethodCase == "CamelCase") {
    toCamelCase(AccessorPrefix);
    toCamelCase(FieldBaseNameLocal);
    AccessorName = AccessorPrefix + FieldBaseNameLocal;
    return;
  }
  if (MethodCase == "camel_Snake_Back") {
    toLower(AccessorPrefix);
    toCamelCase(FieldBaseNameLocal);
    AccessorName = AccessorPrefix + '_' + FieldBaseNameLocal;
    return;
  }
  if (MethodCase == "Camel_Snake_Case") {
    toCamelCase(AccessorPrefix);
    toCamelCase(FieldBaseNameLocal);
    AccessorName = AccessorPrefix + '_' + FieldBaseNameLocal;
    return;
  }
  if (MethodCase == "Leading_upper_snake_case") {
    toCamelCase(AccessorPrefix);
    toLower(FieldBaseNameLocal);
    AccessorName = AccessorPrefix + '_' + FieldBaseNameLocal;
    return;
  }
  FieldBaseNameLocal[0] = llvm::toUpper(FieldBaseNameLocal[0]);
  AccessorName = AccessorPrefix + FieldBaseNameLocal;
}

void GenerateAccessorBase::toLower(std::string &s) const {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return llvm::toLower(c); });
}

void GenerateAccessorBase::toUpper(std::string &s) const {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return llvm::toUpper(c); });
}

void GenerateAccessorBase::toCamelCase(std::string &s) const {
  s[0] = llvm::toUpper(s[0]);
  std::transform(s.begin() + 1, s.end(), s.begin() + 1,
                 [](unsigned char c) { return llvm::toLower(c); });
}

} // namespace clangd
} // namespace clang