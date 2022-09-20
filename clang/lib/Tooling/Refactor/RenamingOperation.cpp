//===--- RenamingOperation.cpp - ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/RenamingOperation.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Refactor/SymbolOperation.h"

using namespace clang;

/// \brief Lexes the given name string.
///
/// \return False if the name was consumed fully, true otherwise.
static bool lexNameString(StringRef Name, Token &Result,
                          const LangOptions &LangOpts) {
  Lexer Lex(SourceLocation(), LangOpts, Name.data(), Name.data(),
            Name.data() + Name.size());
  return !Lex.LexFromRawLexer(Result);
}

namespace clang {
namespace tooling {
namespace rename {

bool isNewNameValid(const OldSymbolName &NewName, bool IsSymbolObjCSelector,
                    IdentifierTable &IDs, const LangOptions &LangOpts) {
  Token Tok;
  if (IsSymbolObjCSelector) {
    // Check if the name is a valid selector.
    for (const auto &Name : NewName.strings()) {
      // Lex the name and verify that it was fully consumed. Then make sure that
      // it's a valid identifier.
      if (lexNameString(Name, Tok, LangOpts) || !Tok.isAnyIdentifier())
        return false;
    }
    return true;
  }

  for (const auto &Name : NewName.strings()) {
    // Lex the name and verify that it was fully consumed. Then make sure that
    // it's a valid identifier that's also not a language keyword.
    if (lexNameString(Name, Tok, LangOpts) || !Tok.isAnyIdentifier() ||
        !tok::isAnyIdentifier(IDs.get(Name).getTokenID()))
      return false;
  }
  return true;
}

bool isNewNameValid(const OldSymbolName &NewName,
                    const SymbolOperation &Operation, IdentifierTable &IDs,
                    const LangOptions &LangOpts) {
  assert(!Operation.symbols().empty());
  return isNewNameValid(NewName,
                        Operation.symbols().front().ObjCSelector.has_value(),
                        IDs, LangOpts);
}

void determineNewNames(OldSymbolName NewName, const SymbolOperation &Operation,
                       SmallVectorImpl<OldSymbolName> &NewNames,
                       const LangOptions &LangOpts) {
  auto Symbols = Operation.symbols();
  assert(!Symbols.empty());
  NewNames.push_back(std::move(NewName));
  if (const auto *PropertyDecl =
          dyn_cast<ObjCPropertyDecl>(Symbols.front().FoundDecl)) {
    assert(NewNames.front().size() == 1 &&
           "Property's name should have one string only");
    StringRef PropertyName = NewNames.front()[0];
    Symbols = Symbols.drop_front();

    auto AddName = [&](const NamedDecl *D, StringRef Name) {
      assert(Symbols.front().FoundDecl == D && "decl is missing");
      NewNames.push_back(OldSymbolName(Name, LangOpts));
      Symbols = Symbols.drop_front();
    };

    if (!PropertyDecl->hasExplicitGetterName()) {
      if (const auto *Getter = PropertyDecl->getGetterMethodDecl())
        AddName(Getter, PropertyName);
    }
    if (!PropertyDecl->hasExplicitSetterName()) {
      if (const auto *Setter = PropertyDecl->getSetterMethodDecl()) {
        auto SetterName = SelectorTable::constructSetterName(PropertyName);
        AddName(Setter, SetterName);
      }
    }
  }
}

} // end namespace rename
} // end namespace tooling
} // end namespace clang
