//===--- InsertionPoint.cpp - Where should we add new code? ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/InsertionPoint.h"
#include "support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceManager.h"
#include <optional>

namespace clang {
namespace clangd {
namespace {

// Choose the decl to insert before, according to an anchor.
// Nullptr means insert at end of DC.
// std::nullopt means no valid place to insert.
std::optional<const Decl *> insertionDecl(const DeclContext &DC,
                                          const Anchor &A) {
  bool LastMatched = false;
  bool ReturnNext = false;
  for (const auto *D : DC.decls()) {
    if (D->isImplicit())
      continue;
    if (ReturnNext)
      return D;

    const Decl *NonTemplate = D;
    if (auto *TD = llvm::dyn_cast<TemplateDecl>(D))
      NonTemplate = TD->getTemplatedDecl();
    bool Matches = A.Match(NonTemplate);
    dlog("    {0} {1} {2}", Matches, D->getDeclKindName(), D);

    switch (A.Direction) {
    case Anchor::Above:
      if (Matches && !LastMatched) {
        // Special case: if "above" matches an access specifier, we actually
        // want to insert below it!
        if (llvm::isa<AccessSpecDecl>(D)) {
          ReturnNext = true;
          continue;
        }
        return D;
      }
      break;
    case Anchor::Below:
      if (LastMatched && !Matches)
        return D;
      break;
    }

    LastMatched = Matches;
  }
  if (ReturnNext || (LastMatched && A.Direction == Anchor::Below))
    return nullptr;
  return std::nullopt;
}

SourceLocation beginLoc(const Decl &D) {
  auto Loc = D.getBeginLoc();
  if (RawComment *Comment = D.getASTContext().getRawCommentForDeclNoCache(&D)) {
    auto CommentLoc = Comment->getBeginLoc();
    if (CommentLoc.isValid() && Loc.isValid() &&
        D.getASTContext().getSourceManager().isBeforeInTranslationUnit(
            CommentLoc, Loc))
      Loc = CommentLoc;
  }
  return Loc;
}

bool any(const Decl *D) { return true; }

SourceLocation endLoc(const DeclContext &DC) {
  const Decl *D = llvm::cast<Decl>(&DC);
  if (auto *OCD = llvm::dyn_cast<ObjCContainerDecl>(D))
    return OCD->getAtEndRange().getBegin();
  return D->getEndLoc();
}

AccessSpecifier getAccessAtEnd(const CXXRecordDecl &C) {
  AccessSpecifier Spec =
      (C.getTagKind() == TagTypeKind::Class ? AS_private : AS_public);
  for (const auto *D : C.decls())
    if (const auto *ASD = llvm::dyn_cast<AccessSpecDecl>(D))
      Spec = ASD->getAccess();
  return Spec;
}

} // namespace

SourceLocation insertionPoint(const DeclContext &DC,
                              llvm::ArrayRef<Anchor> Anchors) {
  dlog("Looking for insertion point in {0}", DC.getDeclKindName());
  for (const auto &A : Anchors) {
    dlog("  anchor ({0})", A.Direction == Anchor::Above ? "above" : "below");
    if (auto D = insertionDecl(DC, A)) {
      dlog("  anchor matched before {0}", *D);
      return *D ? beginLoc(**D) : endLoc(DC);
    }
  }
  dlog("no anchor matched");
  return SourceLocation();
}

llvm::Expected<tooling::Replacement>
insertDecl(llvm::StringRef Code, const DeclContext &DC,
           llvm::ArrayRef<Anchor> Anchors) {
  auto Loc = insertionPoint(DC, Anchors);
  // Fallback: insert at the end.
  if (Loc.isInvalid())
    Loc = endLoc(DC);
  const auto &SM = DC.getParentASTContext().getSourceManager();
  if (!SM.isWrittenInSameFile(Loc, cast<Decl>(DC).getLocation()))
    return error("{0} body in wrong file: {1}", DC.getDeclKindName(),
                 Loc.printToString(SM));
  return tooling::Replacement(SM, Loc, 0, Code);
}

SourceLocation insertionPoint(const CXXRecordDecl &InClass,
                              std::vector<Anchor> Anchors,
                              AccessSpecifier Protection) {
  for (auto &A : Anchors)
    A.Match = [Inner(std::move(A.Match)), Protection](const Decl *D) {
      return D->getAccess() == Protection && Inner(D);
    };
  return insertionPoint(InClass, Anchors);
}

llvm::Expected<tooling::Replacement> insertDecl(llvm::StringRef Code,
                                                const CXXRecordDecl &InClass,
                                                std::vector<Anchor> Anchors,
                                                AccessSpecifier Protection) {
  // Fallback: insert at the bottom of the relevant access section.
  Anchors.push_back({any, Anchor::Below});
  auto Loc = insertionPoint(InClass, std::move(Anchors), Protection);
  std::string CodeBuffer;
  auto &SM = InClass.getASTContext().getSourceManager();
  // Fallback: insert at the end of the class. Check if protection matches!
  if (Loc.isInvalid()) {
    Loc = InClass.getBraceRange().getEnd();
    if (Protection != getAccessAtEnd(InClass)) {
      CodeBuffer = (getAccessSpelling(Protection) + ":\n" + Code).str();
      Code = CodeBuffer;
    }
  }
  if (!SM.isWrittenInSameFile(Loc, InClass.getLocation()))
    return error("Class body in wrong file: {0}", Loc.printToString(SM));
  return tooling::Replacement(SM, Loc, 0, Code);
}

} // namespace clangd
} // namespace clang
