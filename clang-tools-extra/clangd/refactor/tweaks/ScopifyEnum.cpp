//===--- ScopifyEnum.cpp --------------------------------------- -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ParsedAST.h"
#include "Protocol.h"
#include "Selection.h"
#include "SourceCode.h"
#include "XRefs.h"
#include "refactor/Tweak.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace clang::clangd {
namespace {

/// Turns an unscoped into a scoped enum type.
/// Before:
///   enum E { EV1, EV2 };
///        ^
///   void f() { E e1 = EV1; }
///
/// After:
///   enum class E { V1, V2 };
///   void f() { E e1 = E::V1; }
///
/// Note that the respective project code might not compile anymore
/// if it made use of the now-gone implicit conversion to int.
/// This is out of scope for this tweak.

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
  llvm::Error scopifyEnumValue(const EnumConstantDecl &CD, StringRef EnumName,
                               bool StripPrefix);
  llvm::Expected<StringRef> getContentForFile(StringRef FilePath);
  llvm::Error addReplacementForReference(const ReferencesResult::Reference &Ref,
                                         const MakeReplacement &GetReplacement);
  llvm::Error addReplacement(StringRef FilePath, StringRef Content,
                             const tooling::Replacement &Replacement);

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
    return std::move(Err);
  if (auto Err = scopifyEnumValues())
    return std::move(Err);

  return E;
}

llvm::Error ScopifyEnum::addClassKeywordToDeclarations() {
  for (const auto &Ref :
       findReferences(*S->AST, sourceLocToPosition(*SM, D->getBeginLoc()), 0,
                      S->Index, false)
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
  StringRef EnumName(D->getName());
  bool StripPrefix = true;
  for (const EnumConstantDecl *E : D->enumerators()) {
    if (!E->getName().starts_with(EnumName)) {
      StripPrefix = false;
      break;
    }
  }
  for (const EnumConstantDecl *E : D->enumerators()) {
    if (auto Err = scopifyEnumValue(*E, EnumName, StripPrefix))
      return Err;
  }
  return llvm::Error::success();
}

llvm::Error ScopifyEnum::scopifyEnumValue(const EnumConstantDecl &CD,
                                          StringRef EnumName,
                                          bool StripPrefix) {
  for (const auto &Ref :
       findReferences(*S->AST, sourceLocToPosition(*SM, CD.getBeginLoc()), 0,
                      S->Index, false)
           .References) {
    if (Ref.Attributes & ReferencesResult::Declaration) {
      if (StripPrefix) {
        const auto MakeReplacement = [&EnumName](StringRef FilePath,
                                                 StringRef Content,
                                                 unsigned Offset) {
          unsigned Length = EnumName.size();
          if (Content[Offset + Length] == '_')
            ++Length;
          return tooling::Replacement(FilePath, Offset, Length, {});
        };
        if (auto Err = addReplacementForReference(Ref, MakeReplacement))
          return Err;
      }
      continue;
    }

    const auto MakeReplacement = [&](StringRef FilePath, StringRef Content,
                                     unsigned Offset) {
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
      if (StripPrefix) {
        const int ExtraLength =
            Content[Offset + EnumName.size()] == '_' ? 1 : 0;
        if (IsAlreadyScoped())
          return tooling::Replacement(FilePath, Offset,
                                      EnumName.size() + ExtraLength, {});
        return tooling::Replacement(FilePath, Offset + EnumName.size(),
                                    ExtraLength, "::");
      }
      return IsAlreadyScoped() ? tooling::Replacement()
                               : tooling::Replacement(FilePath, Offset, 0,
                                                      EnumName.str() + "::");
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

llvm::Error
ScopifyEnum::addReplacementForReference(const ReferencesResult::Reference &Ref,
                                        const MakeReplacement &GetReplacement) {
  StringRef FilePath = Ref.Loc.uri.file();
  llvm::Expected<StringRef> Content = getContentForFile(FilePath);
  if (!Content)
    return Content.takeError();
  llvm::Expected<size_t> Offset =
      positionToOffset(*Content, Ref.Loc.range.start);
  if (!Offset)
    return Offset.takeError();
  tooling::Replacement Replacement =
      GetReplacement(FilePath, *Content, *Offset);
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

} // namespace
} // namespace clang::clangd
