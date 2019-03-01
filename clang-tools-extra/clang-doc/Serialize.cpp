//===-- Serialize.cpp - ClangDoc Serializer ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Serialize.h"
#include "BitcodeWriter.h"
#include "clang/AST/Comment.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA1.h"

using clang::comments::FullComment;

namespace clang {
namespace doc {
namespace serialize {

SymbolID hashUSR(llvm::StringRef USR) {
  return llvm::SHA1::hash(arrayRefFromStringRef(USR));
}

class ClangDocCommentVisitor
    : public ConstCommentVisitor<ClangDocCommentVisitor> {
public:
  ClangDocCommentVisitor(CommentInfo &CI) : CurrentCI(CI) {}

  void parseComment(const comments::Comment *C);

  void visitTextComment(const TextComment *C);
  void visitInlineCommandComment(const InlineCommandComment *C);
  void visitHTMLStartTagComment(const HTMLStartTagComment *C);
  void visitHTMLEndTagComment(const HTMLEndTagComment *C);
  void visitBlockCommandComment(const BlockCommandComment *C);
  void visitParamCommandComment(const ParamCommandComment *C);
  void visitTParamCommandComment(const TParamCommandComment *C);
  void visitVerbatimBlockComment(const VerbatimBlockComment *C);
  void visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C);
  void visitVerbatimLineComment(const VerbatimLineComment *C);

private:
  std::string getCommandName(unsigned CommandID) const;
  bool isWhitespaceOnly(StringRef S) const;

  CommentInfo &CurrentCI;
};

void ClangDocCommentVisitor::parseComment(const comments::Comment *C) {
  CurrentCI.Kind = C->getCommentKindName();
  ConstCommentVisitor<ClangDocCommentVisitor>::visit(C);
  for (comments::Comment *Child :
       llvm::make_range(C->child_begin(), C->child_end())) {
    CurrentCI.Children.emplace_back(llvm::make_unique<CommentInfo>());
    ClangDocCommentVisitor Visitor(*CurrentCI.Children.back());
    Visitor.parseComment(Child);
  }
}

void ClangDocCommentVisitor::visitTextComment(const TextComment *C) {
  if (!isWhitespaceOnly(C->getText()))
    CurrentCI.Text = C->getText();
}

void ClangDocCommentVisitor::visitInlineCommandComment(
    const InlineCommandComment *C) {
  CurrentCI.Name = getCommandName(C->getCommandID());
  for (unsigned I = 0, E = C->getNumArgs(); I != E; ++I)
    CurrentCI.Args.push_back(C->getArgText(I));
}

void ClangDocCommentVisitor::visitHTMLStartTagComment(
    const HTMLStartTagComment *C) {
  CurrentCI.Name = C->getTagName();
  CurrentCI.SelfClosing = C->isSelfClosing();
  for (unsigned I = 0, E = C->getNumAttrs(); I < E; ++I) {
    const HTMLStartTagComment::Attribute &Attr = C->getAttr(I);
    CurrentCI.AttrKeys.push_back(Attr.Name);
    CurrentCI.AttrValues.push_back(Attr.Value);
  }
}

void ClangDocCommentVisitor::visitHTMLEndTagComment(
    const HTMLEndTagComment *C) {
  CurrentCI.Name = C->getTagName();
  CurrentCI.SelfClosing = true;
}

void ClangDocCommentVisitor::visitBlockCommandComment(
    const BlockCommandComment *C) {
  CurrentCI.Name = getCommandName(C->getCommandID());
  for (unsigned I = 0, E = C->getNumArgs(); I < E; ++I)
    CurrentCI.Args.push_back(C->getArgText(I));
}

void ClangDocCommentVisitor::visitParamCommandComment(
    const ParamCommandComment *C) {
  CurrentCI.Direction =
      ParamCommandComment::getDirectionAsString(C->getDirection());
  CurrentCI.Explicit = C->isDirectionExplicit();
  if (C->hasParamName())
    CurrentCI.ParamName = C->getParamNameAsWritten();
}

void ClangDocCommentVisitor::visitTParamCommandComment(
    const TParamCommandComment *C) {
  if (C->hasParamName())
    CurrentCI.ParamName = C->getParamNameAsWritten();
}

void ClangDocCommentVisitor::visitVerbatimBlockComment(
    const VerbatimBlockComment *C) {
  CurrentCI.Name = getCommandName(C->getCommandID());
  CurrentCI.CloseName = C->getCloseName();
}

void ClangDocCommentVisitor::visitVerbatimBlockLineComment(
    const VerbatimBlockLineComment *C) {
  if (!isWhitespaceOnly(C->getText()))
    CurrentCI.Text = C->getText();
}

void ClangDocCommentVisitor::visitVerbatimLineComment(
    const VerbatimLineComment *C) {
  if (!isWhitespaceOnly(C->getText()))
    CurrentCI.Text = C->getText();
}

bool ClangDocCommentVisitor::isWhitespaceOnly(llvm::StringRef S) const {
  return std::all_of(S.begin(), S.end(), isspace);
}

std::string ClangDocCommentVisitor::getCommandName(unsigned CommandID) const {
  const CommandInfo *Info = CommandTraits::getBuiltinCommandInfo(CommandID);
  if (Info)
    return Info->Name;
  // TODO: Add parsing for \file command.
  return "<not a builtin command>";
}

// Serializing functions.

template <typename T> static std::string serialize(T &I) {
  SmallString<2048> Buffer;
  llvm::BitstreamWriter Stream(Buffer);
  ClangDocBitcodeWriter Writer(Stream);
  Writer.emitBlock(I);
  return Buffer.str().str();
}

std::string serialize(std::unique_ptr<Info> &I) {
  switch (I->IT) {
  case InfoType::IT_namespace:
    return serialize(*static_cast<NamespaceInfo *>(I.get()));
  case InfoType::IT_record:
    return serialize(*static_cast<RecordInfo *>(I.get()));
  case InfoType::IT_enum:
    return serialize(*static_cast<EnumInfo *>(I.get()));
  case InfoType::IT_function:
    return serialize(*static_cast<FunctionInfo *>(I.get()));
  default:
    return "";
  }
}

static void parseFullComment(const FullComment *C, CommentInfo &CI) {
  ClangDocCommentVisitor Visitor(CI);
  Visitor.parseComment(C);
}

static SymbolID getUSRForDecl(const Decl *D) {
  llvm::SmallString<128> USR;
  if (index::generateUSRForDecl(D, USR))
    return SymbolID();
  return hashUSR(USR);
}

static RecordDecl *getDeclForType(const QualType &T) {
  auto *Ty = T->getAs<RecordType>();
  if (!Ty)
    return nullptr;
  return Ty->getDecl()->getDefinition();
}

static bool isPublic(const clang::AccessSpecifier AS,
                     const clang::Linkage Link) {
  if (AS == clang::AccessSpecifier::AS_private)
    return false;
  else if ((Link == clang::Linkage::ModuleLinkage) ||
           (Link == clang::Linkage::ExternalLinkage))
    return true;
  return false; // otherwise, linkage is some form of internal linkage
}

static void parseFields(RecordInfo &I, const RecordDecl *D, bool PublicOnly) {
  for (const FieldDecl *F : D->fields()) {
    if (PublicOnly && !isPublic(F->getAccessUnsafe(), F->getLinkageInternal()))
      continue;
    if (const auto *T = getDeclForType(F->getTypeSourceInfo()->getType())) {
      // Use getAccessUnsafe so that we just get the default AS_none if it's not
      // valid, as opposed to an assert.
      if (const auto *N = dyn_cast<EnumDecl>(T)) {
        I.Members.emplace_back(getUSRForDecl(T), N->getNameAsString(),
                               InfoType::IT_enum, F->getNameAsString(),
                               N->getAccessUnsafe());
        continue;
      } else if (const auto *N = dyn_cast<RecordDecl>(T)) {
        I.Members.emplace_back(getUSRForDecl(T), N->getNameAsString(),
                               InfoType::IT_record, F->getNameAsString(),
                               N->getAccessUnsafe());
        continue;
      }
    }
    I.Members.emplace_back(F->getTypeSourceInfo()->getType().getAsString(),
                           F->getNameAsString(), F->getAccessUnsafe());
  }
}

static void parseEnumerators(EnumInfo &I, const EnumDecl *D) {
  for (const EnumConstantDecl *E : D->enumerators())
    I.Members.emplace_back(E->getNameAsString());
}

static void parseParameters(FunctionInfo &I, const FunctionDecl *D) {
  for (const ParmVarDecl *P : D->parameters()) {
    if (const auto *T = getDeclForType(P->getOriginalType())) {
      if (const auto *N = dyn_cast<EnumDecl>(T)) {
        I.Params.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_enum, P->getNameAsString());
        continue;
      } else if (const auto *N = dyn_cast<RecordDecl>(T)) {
        I.Params.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_record, P->getNameAsString());
        continue;
      }
    }
    I.Params.emplace_back(P->getOriginalType().getAsString(),
                          P->getNameAsString());
  }
}

static void parseBases(RecordInfo &I, const CXXRecordDecl *D) {
  // Don't parse bases if this isn't a definition.
  if (!D->isThisDeclarationADefinition())
    return;
  for (const CXXBaseSpecifier &B : D->bases()) {
    if (B.isVirtual())
      continue;
    if (const auto *P = getDeclForType(B.getType()))
      I.Parents.emplace_back(getUSRForDecl(P), P->getNameAsString(),
                             InfoType::IT_record);
    else
      I.Parents.emplace_back(B.getType().getAsString());
  }
  for (const CXXBaseSpecifier &B : D->vbases()) {
    if (const auto *P = getDeclForType(B.getType()))
      I.VirtualParents.emplace_back(getUSRForDecl(P), P->getNameAsString(),
                                    InfoType::IT_record);
    else
      I.VirtualParents.emplace_back(B.getType().getAsString());
  }
}

template <typename T>
static void
populateParentNamespaces(llvm::SmallVector<Reference, 4> &Namespaces,
                         const T *D) {
  const auto *DC = dyn_cast<DeclContext>(D);
  while ((DC = DC->getParent())) {
    if (const auto *N = dyn_cast<NamespaceDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_namespace);
    else if (const auto *N = dyn_cast<RecordDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_record);
    else if (const auto *N = dyn_cast<FunctionDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_function);
    else if (const auto *N = dyn_cast<EnumDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_enum);
  }
}

template <typename T>
static void populateInfo(Info &I, const T *D, const FullComment *C) {
  I.USR = getUSRForDecl(D);
  I.Name = D->getNameAsString();
  populateParentNamespaces(I.Namespace, D);
  if (C) {
    I.Description.emplace_back();
    parseFullComment(C, I.Description.back());
  }
}

template <typename T>
static void populateSymbolInfo(SymbolInfo &I, const T *D, const FullComment *C,
                               int LineNumber, StringRef Filename) {
  populateInfo(I, D, C);
  if (D->isThisDeclarationADefinition())
    I.DefLoc.emplace(LineNumber, Filename);
  else
    I.Loc.emplace_back(LineNumber, Filename);
}

static void populateFunctionInfo(FunctionInfo &I, const FunctionDecl *D,
                                 const FullComment *FC, int LineNumber,
                                 StringRef Filename) {
  populateSymbolInfo(I, D, FC, LineNumber, Filename);
  if (const auto *T = getDeclForType(D->getReturnType())) {
    if (dyn_cast<EnumDecl>(T))
      I.ReturnType =
          TypeInfo(getUSRForDecl(T), T->getNameAsString(), InfoType::IT_enum);
    else if (dyn_cast<RecordDecl>(T))
      I.ReturnType =
          TypeInfo(getUSRForDecl(T), T->getNameAsString(), InfoType::IT_record);
  } else {
    I.ReturnType = TypeInfo(D->getReturnType().getAsString());
  }
  parseParameters(I, D);
}

std::unique_ptr<Info> emitInfo(const NamespaceDecl *D, const FullComment *FC,
                               int LineNumber, llvm::StringRef File,
                               bool PublicOnly) {
  if (PublicOnly && ((D->isAnonymousNamespace()) ||
                     !isPublic(D->getAccess(), D->getLinkageInternal())))
    return nullptr;
  auto I = llvm::make_unique<NamespaceInfo>();
  populateInfo(*I, D, FC);
  return std::unique_ptr<Info>{std::move(I)};
}

std::unique_ptr<Info> emitInfo(const RecordDecl *D, const FullComment *FC,
                               int LineNumber, llvm::StringRef File,
                               bool PublicOnly) {
  if (PublicOnly && !isPublic(D->getAccess(), D->getLinkageInternal()))
    return nullptr;
  auto I = llvm::make_unique<RecordInfo>();
  populateSymbolInfo(*I, D, FC, LineNumber, File);
  I->TagType = D->getTagKind();
  parseFields(*I, D, PublicOnly);
  if (const auto *C = dyn_cast<CXXRecordDecl>(D))
    parseBases(*I, C);
  return std::unique_ptr<Info>{std::move(I)};
}

std::unique_ptr<Info> emitInfo(const FunctionDecl *D, const FullComment *FC,
                               int LineNumber, llvm::StringRef File,
                               bool PublicOnly) {
  if (PublicOnly && !isPublic(D->getAccess(), D->getLinkageInternal()))
    return nullptr;
  FunctionInfo Func;
  populateFunctionInfo(Func, D, FC, LineNumber, File);
  Func.Access = clang::AccessSpecifier::AS_none;

  // Wrap in enclosing scope
  auto I = llvm::make_unique<NamespaceInfo>();
  if (!Func.Namespace.empty())
    I->USR = Func.Namespace[0].USR;
  else
    I->USR = SymbolID();
  I->ChildFunctions.emplace_back(std::move(Func));
  return std::unique_ptr<Info>{std::move(I)};
}

std::unique_ptr<Info> emitInfo(const CXXMethodDecl *D, const FullComment *FC,
                               int LineNumber, llvm::StringRef File,
                               bool PublicOnly) {
  if (PublicOnly && !isPublic(D->getAccess(), D->getLinkageInternal()))
    return nullptr;
  FunctionInfo Func;
  populateFunctionInfo(Func, D, FC, LineNumber, File);
  Func.IsMethod = true;

  SymbolID ParentUSR = getUSRForDecl(D->getParent());
  Func.Parent = Reference{ParentUSR, D->getParent()->getNameAsString(),
                          InfoType::IT_record};
  Func.Access = D->getAccess();

  // Wrap in enclosing scope
  auto I = llvm::make_unique<RecordInfo>();
  I->USR = ParentUSR;
  I->ChildFunctions.emplace_back(std::move(Func));
  return std::unique_ptr<Info>{std::move(I)};
}

std::unique_ptr<Info> emitInfo(const EnumDecl *D, const FullComment *FC,
                               int LineNumber, llvm::StringRef File,
                               bool PublicOnly) {
  if (PublicOnly && !isPublic(D->getAccess(), D->getLinkageInternal()))
    return nullptr;
  EnumInfo Enum;
  populateSymbolInfo(Enum, D, FC, LineNumber, File);
  Enum.Scoped = D->isScoped();
  parseEnumerators(Enum, D);

  // Wrap in enclosing scope
  if (!Enum.Namespace.empty()) {
    switch (Enum.Namespace[0].RefType) {
    case InfoType::IT_namespace: {
      auto I = llvm::make_unique<NamespaceInfo>();
      I->USR = Enum.Namespace[0].USR;
      I->ChildEnums.emplace_back(std::move(Enum));
      return std::unique_ptr<Info>{std::move(I)};
    }
    case InfoType::IT_record: {
      auto I = llvm::make_unique<RecordInfo>();
      I->USR = Enum.Namespace[0].USR;
      I->ChildEnums.emplace_back(std::move(Enum));
      return std::unique_ptr<Info>{std::move(I)};
    }
    default:
      break;
    }
  }

  // Put in global namespace
  auto I = llvm::make_unique<NamespaceInfo>();
  I->USR = SymbolID();
  I->ChildEnums.emplace_back(std::move(Enum));
  return std::unique_ptr<Info>{std::move(I)};
}

} // namespace serialize
} // namespace doc
} // namespace clang
