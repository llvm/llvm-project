//===--- ScopifyEnum.cpp --------------------------------------- -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ParsedAST.h"
#include "XRefs.h"
#include "refactor/Tweak.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <functional>

namespace clang::clangd {
namespace {

/// Turns an unscoped into a scoped enum type.
/// Before:
///   enum E { EV1, EV2 };
///        ^
///   void f() { E e1 = EV1; }
///
/// After:
///   enum class E { EV1, EV2 };
///   void f() { E e1 = E::EV1; }
///
/// Note that the respective project code might not compile anymore
/// if it made use of the now-gone implicit conversion to int.
/// This is out of scope for this tweak.
///
/// TODO: In the above example, we could detect that the values
///       start with the enum name, and remove that prefix.

class ScopifyEnum : public Tweak {
  const char *id() const final;
  std::string title() const override { return "Convert to scoped enum"; }
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }
  bool prepare(const Selection &Inputs) override;
  Expected<Tweak::Effect> apply(const Selection &Inputs) override;

  using MakeReplacement =
      std::function<tooling::Replacement(StringRef, StringRef, unsigned)>;
  llvm::Error addClassKeywordToDeclarations();
  llvm::Error scopifyEnumValues();
  llvm::Error scopifyEnumValue(const EnumConstantDecl &CD, StringRef Prefix);
  llvm::Expected<StringRef> getContentForFile(StringRef FilePath);
  unsigned getOffsetFromPosition(const Position &Pos, StringRef Content) const;
  llvm::Error addReplacementForReference(const ReferencesResult::Reference &Ref,
                                         const MakeReplacement &GetReplacement);
  llvm::Error addReplacement(StringRef FilePath, StringRef Content,
                             const tooling::Replacement &Replacement);
  Position getPosition(const Decl &D) const;

  const EnumDecl *D = nullptr;
  const Selection *S = nullptr;
  SourceManager *SM = nullptr;
  llvm::SmallVector<std::unique_ptr<llvm::MemoryBuffer>> ExtraBuffers;
  llvm::StringMap<StringRef> ContentPerFile;
  Effect E;
};

REGISTER_TWEAK(ScopifyEnum)

bool ScopifyEnum::prepare(const Selection &Inputs) {
  if (!Inputs.AST->getLangOpts().CPlusPlus11)
    return false;
  const SelectionTree::Node *N = Inputs.ASTSelection.commonAncestor();
  if (!N)
    return false;
  D = N->ASTNode.get<EnumDecl>();
  return D && !D->isScoped() && D->isThisDeclarationADefinition();
}

Expected<Tweak::Effect> ScopifyEnum::apply(const Selection &Inputs) {
  S = &Inputs;
  SM = &S->AST->getSourceManager();
  E.FormatEdits = false;
  ContentPerFile.insert(std::make_pair(SM->getFilename(D->getLocation()),
                                       SM->getBufferData(SM->getMainFileID())));

  if (auto Err = addClassKeywordToDeclarations())
    return Err;
  if (auto Err = scopifyEnumValues())
    return Err;

  return E;
}

llvm::Error ScopifyEnum::addClassKeywordToDeclarations() {
  for (const auto &Ref :
       findReferences(*S->AST, getPosition(*D), 0, S->Index, false)
           .References) {
    if (!(Ref.Attributes & ReferencesResult::Declaration))
      continue;

    static const auto MakeReplacement = [](StringRef FilePath,
                                           StringRef Content, unsigned Offset) {
      return tooling::Replacement(FilePath, Offset, 0, "class ");
    };
    if (auto Err = addReplacementForReference(Ref, MakeReplacement))
      return Err;
  }
  return llvm::Error::success();
}

llvm::Error ScopifyEnum::scopifyEnumValues() {
  std::string PrefixToInsert(D->getName());
  PrefixToInsert += "::";
  for (auto E : D->enumerators()) {
    if (auto Err = scopifyEnumValue(*E, PrefixToInsert))
      return Err;
  }
  return llvm::Error::success();
}

llvm::Error ScopifyEnum::scopifyEnumValue(const EnumConstantDecl &CD,
                                          StringRef Prefix) {
  for (const auto &Ref :
       findReferences(*S->AST, getPosition(CD), 0, S->Index, false)
           .References) {
    if (Ref.Attributes & ReferencesResult::Declaration)
      continue;

    const auto MakeReplacement = [&Prefix](StringRef FilePath,
                                           StringRef Content, unsigned Offset) {
      const auto IsAlreadyScoped = [Content, Offset] {
        if (Offset < 2)
          return false;
        unsigned I = Offset;
        while (--I > 0) {
          switch (Content[I]) {
          case ' ':
          case '\t':
          case '\n':
            continue;
          case ':':
            if (Content[I - 1] == ':')
              return true;
            [[fallthrough]];
          default:
            return false;
          }
        }
        return false;
      };
      return IsAlreadyScoped()
                 ? tooling::Replacement()
                 : tooling::Replacement(FilePath, Offset, 0, Prefix);
    };
    if (auto Err = addReplacementForReference(Ref, MakeReplacement))
      return Err;
  }

  return llvm::Error::success();
}

llvm::Expected<StringRef> ScopifyEnum::getContentForFile(StringRef FilePath) {
  if (auto It = ContentPerFile.find(FilePath); It != ContentPerFile.end())
    return It->second;
  auto Buffer = S->FS->getBufferForFile(FilePath);
  if (!Buffer)
    return llvm::errorCodeToError(Buffer.getError());
  StringRef Content = Buffer->get()->getBuffer();
  ExtraBuffers.push_back(std::move(*Buffer));
  ContentPerFile.insert(std::make_pair(FilePath, Content));
  return Content;
}

unsigned int ScopifyEnum::getOffsetFromPosition(const Position &Pos,
                                                StringRef Content) const {
  unsigned int Offset = 0;

  for (std::size_t LinesRemaining = Pos.line;
       Offset < Content.size() && LinesRemaining;) {
    if (Content[Offset++] == '\n')
      --LinesRemaining;
  }
  return Offset + Pos.character;
}

llvm::Error
ScopifyEnum::addReplacementForReference(const ReferencesResult::Reference &Ref,
                                        const MakeReplacement &GetReplacement) {
  StringRef FilePath = Ref.Loc.uri.file();
  auto Content = getContentForFile(FilePath);
  if (!Content)
    return Content.takeError();
  unsigned Offset = getOffsetFromPosition(Ref.Loc.range.start, *Content);
  tooling::Replacement Replacement = GetReplacement(FilePath, *Content, Offset);
  if (Replacement.isApplicable())
    return addReplacement(FilePath, *Content, Replacement);
  return llvm::Error::success();
}

llvm::Error
ScopifyEnum::addReplacement(StringRef FilePath, StringRef Content,
                            const tooling::Replacement &Replacement) {
  Edit &TheEdit = E.ApplyEdits[FilePath];
  TheEdit.InitialCode = Content;
  if (auto Err = TheEdit.Replacements.add(Replacement))
    return Err;
  return llvm::Error::success();
}

Position ScopifyEnum::getPosition(const Decl &D) const {
  const SourceLocation Loc = D.getLocation();
  Position Pos;
  Pos.line = SM->getSpellingLineNumber(Loc) - 1;
  Pos.character = SM->getSpellingColumnNumber(Loc) - 1;
  return Pos;
}

} // namespace
} // namespace clang::clangd
