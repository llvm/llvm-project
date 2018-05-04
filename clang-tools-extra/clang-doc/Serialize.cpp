//===-- Serializer.cpp - ClangDoc Serializer --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

static void parseFields(RecordInfo &I, const RecordDecl *D) {
  for (const FieldDecl *F : D->fields()) {
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

std::string emitInfo(const NamespaceDecl *D, const FullComment *FC,
                     int LineNumber, llvm::StringRef File) {
  NamespaceInfo I;
  populateInfo(I, D, FC);
  return serialize(I);
}

std::string emitInfo(const RecordDecl *D, const FullComment *FC, int LineNumber,
                     llvm::StringRef File) {
  RecordInfo I;
  populateSymbolInfo(I, D, FC, LineNumber, File);
  I.TagType = D->getTagKind();
  parseFields(I, D);
  if (const auto *C = dyn_cast<CXXRecordDecl>(D))
    parseBases(I, C);
  return serialize(I);
}

std::string emitInfo(const FunctionDecl *D, const FullComment *FC,
                     int LineNumber, llvm::StringRef File) {
  FunctionInfo I;
  populateFunctionInfo(I, D, FC, LineNumber, File);
  I.Access = clang::AccessSpecifier::AS_none;
  return serialize(I);
}

std::string emitInfo(const CXXMethodDecl *D, const FullComment *FC,
                     int LineNumber, llvm::StringRef File) {
  FunctionInfo I;
  populateFunctionInfo(I, D, FC, LineNumber, File);
  I.IsMethod = true;
  I.Parent = Reference{getUSRForDecl(D->getParent()),
                       D->getParent()->getNameAsString(), InfoType::IT_record};
  I.Access = D->getAccess();
  return serialize(I);
}

std::string emitInfo(const EnumDecl *D, const FullComment *FC, int LineNumber,
                     llvm::StringRef File) {
  EnumInfo I;
  populateSymbolInfo(I, D, FC, LineNumber, File);
  I.Scoped = D->isScoped();
  parseEnumerators(I, D);
  return serialize(I);
}

} // namespace serialize
} // namespace doc
} // namespace clang
