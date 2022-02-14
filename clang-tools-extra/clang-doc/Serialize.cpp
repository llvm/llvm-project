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

template <typename T>
static void
populateParentNamespaces(llvm::SmallVector<Reference, 4> &Namespaces,
                         const T *D, bool &IsAnonymousNamespace);

// A function to extract the appropriate relative path for a given info's
// documentation. The path returned is a composite of the parent namespaces.
//
// Example: Given the below, the directory path for class C info will be
// <root>/A/B
//
// namespace A {
// namespace B {
//
// class C {};
//
// }
// }
llvm::SmallString<128>
getInfoRelativePath(const llvm::SmallVectorImpl<doc::Reference> &Namespaces) {
  llvm::SmallString<128> Path;
  for (auto R = Namespaces.rbegin(), E = Namespaces.rend(); R != E; ++R)
    llvm::sys::path::append(Path, R->Name);
  return Path;
}

llvm::SmallString<128> getInfoRelativePath(const Decl *D) {
  llvm::SmallVector<Reference, 4> Namespaces;
  // The third arg in populateParentNamespaces is a boolean passed by reference,
  // its value is not relevant in here so it's not used anywhere besides the
  // function call
  bool B = true;
  populateParentNamespaces(Namespaces, D, B);
  return getInfoRelativePath(Namespaces);
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
    CurrentCI.Children.emplace_back(std::make_unique<CommentInfo>());
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
  if (const RecordDecl *D = T->getAsRecordDecl())
    return D->getDefinition();
  return nullptr;
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

static bool shouldSerializeInfo(bool PublicOnly, bool IsInAnonymousNamespace,
                                const NamedDecl *D) {
  bool IsAnonymousNamespace = false;
  if (const auto *N = dyn_cast<NamespaceDecl>(D))
    IsAnonymousNamespace = N->isAnonymousNamespace();
  return !PublicOnly ||
         (!IsInAnonymousNamespace && !IsAnonymousNamespace &&
          isPublic(D->getAccessUnsafe(), D->getLinkageInternal()));
}

// There are two uses for this function.
// 1) Getting the resulting mode of inheritance of a record.
//    Example: class A {}; class B : private A {}; class C : public B {};
//    It's explicit that C is publicly inherited from C and B is privately
//    inherited from A. It's not explicit but C is also privately inherited from
//    A. This is the AS that this function calculates. FirstAS is the
//    inheritance mode of `class C : B` and SecondAS is the inheritance mode of
//    `class B : A`.
// 2) Getting the inheritance mode of an inherited attribute / method.
//    Example : class A { public: int M; }; class B : private A {};
//    Class B is inherited from class A, which has a public attribute. This
//    attribute is now part of the derived class B but it's not public. This
//    will be private because the inheritance is private. This is the AS that
//    this function calculates. FirstAS is the inheritance mode and SecondAS is
//    the AS of the attribute / method.
static AccessSpecifier getFinalAccessSpecifier(AccessSpecifier FirstAS,
                                               AccessSpecifier SecondAS) {
  if (FirstAS == AccessSpecifier::AS_none ||
      SecondAS == AccessSpecifier::AS_none)
    return AccessSpecifier::AS_none;
  if (FirstAS == AccessSpecifier::AS_private ||
      SecondAS == AccessSpecifier::AS_private)
    return AccessSpecifier::AS_private;
  if (FirstAS == AccessSpecifier::AS_protected ||
      SecondAS == AccessSpecifier::AS_protected)
    return AccessSpecifier::AS_protected;
  return AccessSpecifier::AS_public;
}

// The Access parameter is only provided when parsing the field of an inherited
// record, the access specification of the field depends on the inheritance mode
static void parseFields(RecordInfo &I, const RecordDecl *D, bool PublicOnly,
                        AccessSpecifier Access = AccessSpecifier::AS_public) {
  for (const FieldDecl *F : D->fields()) {
    if (!shouldSerializeInfo(PublicOnly, /*IsInAnonymousNamespace=*/false, F))
      continue;
    if (const auto *T = getDeclForType(F->getTypeSourceInfo()->getType())) {
      // Use getAccessUnsafe so that we just get the default AS_none if it's not
      // valid, as opposed to an assert.
      if (const auto *N = dyn_cast<EnumDecl>(T)) {
        I.Members.emplace_back(
            getUSRForDecl(T), N->getNameAsString(), InfoType::IT_enum,
            getInfoRelativePath(N), F->getNameAsString(),
            getFinalAccessSpecifier(Access, N->getAccessUnsafe()));
        continue;
      } else if (const auto *N = dyn_cast<RecordDecl>(T)) {
        I.Members.emplace_back(
            getUSRForDecl(T), N->getNameAsString(), InfoType::IT_record,
            getInfoRelativePath(N), F->getNameAsString(),
            getFinalAccessSpecifier(Access, N->getAccessUnsafe()));
        continue;
      }
    }
    I.Members.emplace_back(
        F->getTypeSourceInfo()->getType().getAsString(), F->getNameAsString(),
        getFinalAccessSpecifier(Access, F->getAccessUnsafe()));
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
                              InfoType::IT_enum, getInfoRelativePath(N),
                              P->getNameAsString());
        continue;
      } else if (const auto *N = dyn_cast<RecordDecl>(T)) {
        I.Params.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_record, getInfoRelativePath(N),
                              P->getNameAsString());
        continue;
      }
    }
    I.Params.emplace_back(P->getOriginalType().getAsString(),
                          P->getNameAsString());
  }
}

// TODO: Remove the serialization of Parents and VirtualParents, this
// information is also extracted in the other definition of parseBases.
static void parseBases(RecordInfo &I, const CXXRecordDecl *D) {
  // Don't parse bases if this isn't a definition.
  if (!D->isThisDeclarationADefinition())
    return;
  for (const CXXBaseSpecifier &B : D->bases()) {
    if (B.isVirtual())
      continue;
    if (const auto *Ty = B.getType()->getAs<TemplateSpecializationType>()) {
      const TemplateDecl *D = Ty->getTemplateName().getAsTemplateDecl();
      I.Parents.emplace_back(getUSRForDecl(D), B.getType().getAsString(),
                             InfoType::IT_record);
    } else if (const RecordDecl *P = getDeclForType(B.getType()))
      I.Parents.emplace_back(getUSRForDecl(P), P->getNameAsString(),
                             InfoType::IT_record, getInfoRelativePath(P));
    else
      I.Parents.emplace_back(B.getType().getAsString());
  }
  for (const CXXBaseSpecifier &B : D->vbases()) {
    if (const auto *P = getDeclForType(B.getType()))
      I.VirtualParents.emplace_back(getUSRForDecl(P), P->getNameAsString(),
                                    InfoType::IT_record,
                                    getInfoRelativePath(P));
    else
      I.VirtualParents.emplace_back(B.getType().getAsString());
  }
}

template <typename T>
static void
populateParentNamespaces(llvm::SmallVector<Reference, 4> &Namespaces,
                         const T *D, bool &IsInAnonymousNamespace) {
  const auto *DC = cast<DeclContext>(D);
  while ((DC = DC->getParent())) {
    if (const auto *N = dyn_cast<NamespaceDecl>(DC)) {
      std::string Namespace;
      if (N->isAnonymousNamespace()) {
        Namespace = "@nonymous_namespace";
        IsInAnonymousNamespace = true;
      } else
        Namespace = N->getNameAsString();
      Namespaces.emplace_back(getUSRForDecl(N), Namespace,
                              InfoType::IT_namespace);
    } else if (const auto *N = dyn_cast<RecordDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_record);
    else if (const auto *N = dyn_cast<FunctionDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_function);
    else if (const auto *N = dyn_cast<EnumDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_enum);
  }
  // The global namespace should be added to the list of namespaces if the decl
  // corresponds to a Record and if it doesn't have any namespace (because this
  // means it's in the global namespace). Also if its outermost namespace is a
  // record because that record matches the previous condition mentioned.
  if ((Namespaces.empty() && isa<RecordDecl>(D)) ||
      (!Namespaces.empty() && Namespaces.back().RefType == InfoType::IT_record))
    Namespaces.emplace_back(SymbolID(), "GlobalNamespace",
                            InfoType::IT_namespace);
}

template <typename T>
static void populateInfo(Info &I, const T *D, const FullComment *C,
                         bool &IsInAnonymousNamespace) {
  I.USR = getUSRForDecl(D);
  I.Name = D->getNameAsString();
  populateParentNamespaces(I.Namespace, D, IsInAnonymousNamespace);
  if (C) {
    I.Description.emplace_back();
    parseFullComment(C, I.Description.back());
  }
}

template <typename T>
static void populateSymbolInfo(SymbolInfo &I, const T *D, const FullComment *C,
                               int LineNumber, StringRef Filename,
                               bool IsFileInRootDir,
                               bool &IsInAnonymousNamespace) {
  populateInfo(I, D, C, IsInAnonymousNamespace);
  if (D->isThisDeclarationADefinition())
    I.DefLoc.emplace(LineNumber, Filename, IsFileInRootDir);
  else
    I.Loc.emplace_back(LineNumber, Filename, IsFileInRootDir);
}

static void populateFunctionInfo(FunctionInfo &I, const FunctionDecl *D,
                                 const FullComment *FC, int LineNumber,
                                 StringRef Filename, bool IsFileInRootDir,
                                 bool &IsInAnonymousNamespace) {
  populateSymbolInfo(I, D, FC, LineNumber, Filename, IsFileInRootDir,
                     IsInAnonymousNamespace);
  if (const auto *T = getDeclForType(D->getReturnType())) {
    if (isa<EnumDecl>(T))
      I.ReturnType = TypeInfo(getUSRForDecl(T), T->getNameAsString(),
                              InfoType::IT_enum, getInfoRelativePath(T));
    else if (isa<RecordDecl>(T))
      I.ReturnType = TypeInfo(getUSRForDecl(T), T->getNameAsString(),
                              InfoType::IT_record, getInfoRelativePath(T));
  } else {
    I.ReturnType = TypeInfo(D->getReturnType().getAsString());
  }
  parseParameters(I, D);
}

static void
parseBases(RecordInfo &I, const CXXRecordDecl *D, bool IsFileInRootDir,
           bool PublicOnly, bool IsParent,
           AccessSpecifier ParentAccess = AccessSpecifier::AS_public) {
  // Don't parse bases if this isn't a definition.
  if (!D->isThisDeclarationADefinition())
    return;
  for (const CXXBaseSpecifier &B : D->bases()) {
    if (const RecordType *Ty = B.getType()->getAs<RecordType>()) {
      if (const CXXRecordDecl *Base =
              cast_or_null<CXXRecordDecl>(Ty->getDecl()->getDefinition())) {
        // Initialized without USR and name, this will be set in the following
        // if-else stmt.
        BaseRecordInfo BI(
            {}, "", getInfoRelativePath(Base), B.isVirtual(),
            getFinalAccessSpecifier(ParentAccess, B.getAccessSpecifier()),
            IsParent);
        if (const auto *Ty = B.getType()->getAs<TemplateSpecializationType>()) {
          const TemplateDecl *D = Ty->getTemplateName().getAsTemplateDecl();
          BI.USR = getUSRForDecl(D);
          BI.Name = B.getType().getAsString();
        } else {
          BI.USR = getUSRForDecl(Base);
          BI.Name = Base->getNameAsString();
        }
        parseFields(BI, Base, PublicOnly, BI.Access);
        for (const auto &Decl : Base->decls())
          if (const auto *MD = dyn_cast<CXXMethodDecl>(Decl)) {
            // Don't serialize private methods
            if (MD->getAccessUnsafe() == AccessSpecifier::AS_private ||
                !MD->isUserProvided())
              continue;
            FunctionInfo FI;
            FI.IsMethod = true;
            // The seventh arg in populateFunctionInfo is a boolean passed by
            // reference, its value is not relevant in here so it's not used
            // anywhere besides the function call.
            bool IsInAnonymousNamespace;
            populateFunctionInfo(FI, MD, /*FullComment=*/{}, /*LineNumber=*/{},
                                 /*FileName=*/{}, IsFileInRootDir,
                                 IsInAnonymousNamespace);
            FI.Access =
                getFinalAccessSpecifier(BI.Access, MD->getAccessUnsafe());
            BI.ChildFunctions.emplace_back(std::move(FI));
          }
        I.Bases.emplace_back(std::move(BI));
        // Call this function recursively to get the inherited classes of
        // this base; these new bases will also get stored in the original
        // RecordInfo: I.
        parseBases(I, Base, IsFileInRootDir, PublicOnly, false,
                   I.Bases.back().Access);
      }
    }
  }
}

std::pair<std::unique_ptr<Info>, std::unique_ptr<Info>>
emitInfo(const NamespaceDecl *D, const FullComment *FC, int LineNumber,
         llvm::StringRef File, bool IsFileInRootDir, bool PublicOnly) {
  auto I = std::make_unique<NamespaceInfo>();
  bool IsInAnonymousNamespace = false;
  populateInfo(*I, D, FC, IsInAnonymousNamespace);
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  I->Name = D->isAnonymousNamespace()
                ? llvm::SmallString<16>("@nonymous_namespace")
                : I->Name;
  I->Path = getInfoRelativePath(I->Namespace);
  if (I->Namespace.empty() && I->USR == SymbolID())
    return {std::unique_ptr<Info>{std::move(I)}, nullptr};

  auto ParentI = std::make_unique<NamespaceInfo>();
  ParentI->USR = I->Namespace.empty() ? SymbolID() : I->Namespace[0].USR;
  ParentI->ChildNamespaces.emplace_back(I->USR, I->Name, InfoType::IT_namespace,
                                        getInfoRelativePath(I->Namespace));
  if (I->Namespace.empty())
    ParentI->Path = getInfoRelativePath(ParentI->Namespace);
  return {std::unique_ptr<Info>{std::move(I)},
          std::unique_ptr<Info>{std::move(ParentI)}};
}

std::pair<std::unique_ptr<Info>, std::unique_ptr<Info>>
emitInfo(const RecordDecl *D, const FullComment *FC, int LineNumber,
         llvm::StringRef File, bool IsFileInRootDir, bool PublicOnly) {
  auto I = std::make_unique<RecordInfo>();
  bool IsInAnonymousNamespace = false;
  populateSymbolInfo(*I, D, FC, LineNumber, File, IsFileInRootDir,
                     IsInAnonymousNamespace);
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  I->TagType = D->getTagKind();
  parseFields(*I, D, PublicOnly);
  if (const auto *C = dyn_cast<CXXRecordDecl>(D)) {
    if (const TypedefNameDecl *TD = C->getTypedefNameForAnonDecl()) {
      I->Name = TD->getNameAsString();
      I->IsTypeDef = true;
    }
    // TODO: remove first call to parseBases, that function should be deleted
    parseBases(*I, C);
    parseBases(*I, C, IsFileInRootDir, PublicOnly, true);
  }
  I->Path = getInfoRelativePath(I->Namespace);

  switch (I->Namespace[0].RefType) {
  case InfoType::IT_namespace: {
    auto ParentI = std::make_unique<NamespaceInfo>();
    ParentI->USR = I->Namespace[0].USR;
    ParentI->ChildRecords.emplace_back(I->USR, I->Name, InfoType::IT_record,
                                       getInfoRelativePath(I->Namespace));
    return {std::unique_ptr<Info>{std::move(I)},
            std::unique_ptr<Info>{std::move(ParentI)}};
  }
  case InfoType::IT_record: {
    auto ParentI = std::make_unique<RecordInfo>();
    ParentI->USR = I->Namespace[0].USR;
    ParentI->ChildRecords.emplace_back(I->USR, I->Name, InfoType::IT_record,
                                       getInfoRelativePath(I->Namespace));
    return {std::unique_ptr<Info>{std::move(I)},
            std::unique_ptr<Info>{std::move(ParentI)}};
  }
  default:
    llvm_unreachable("Invalid reference type for parent namespace");
  }
}

std::pair<std::unique_ptr<Info>, std::unique_ptr<Info>>
emitInfo(const FunctionDecl *D, const FullComment *FC, int LineNumber,
         llvm::StringRef File, bool IsFileInRootDir, bool PublicOnly) {
  FunctionInfo Func;
  bool IsInAnonymousNamespace = false;
  populateFunctionInfo(Func, D, FC, LineNumber, File, IsFileInRootDir,
                       IsInAnonymousNamespace);
  Func.Access = clang::AccessSpecifier::AS_none;
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  // Wrap in enclosing scope
  auto ParentI = std::make_unique<NamespaceInfo>();
  if (!Func.Namespace.empty())
    ParentI->USR = Func.Namespace[0].USR;
  else
    ParentI->USR = SymbolID();
  if (Func.Namespace.empty())
    ParentI->Path = getInfoRelativePath(ParentI->Namespace);
  ParentI->ChildFunctions.emplace_back(std::move(Func));
  // Info is wrapped in its parent scope so it's returned in the second position
  return {nullptr, std::unique_ptr<Info>{std::move(ParentI)}};
}

std::pair<std::unique_ptr<Info>, std::unique_ptr<Info>>
emitInfo(const CXXMethodDecl *D, const FullComment *FC, int LineNumber,
         llvm::StringRef File, bool IsFileInRootDir, bool PublicOnly) {
  FunctionInfo Func;
  bool IsInAnonymousNamespace = false;
  populateFunctionInfo(Func, D, FC, LineNumber, File, IsFileInRootDir,
                       IsInAnonymousNamespace);
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  Func.IsMethod = true;

  const NamedDecl *Parent = nullptr;
  if (const auto *SD =
          dyn_cast<ClassTemplateSpecializationDecl>(D->getParent()))
    Parent = SD->getSpecializedTemplate();
  else
    Parent = D->getParent();

  SymbolID ParentUSR = getUSRForDecl(Parent);
  Func.Parent =
      Reference{ParentUSR, Parent->getNameAsString(), InfoType::IT_record};
  Func.Access = D->getAccess();

  // Wrap in enclosing scope
  auto ParentI = std::make_unique<RecordInfo>();
  ParentI->USR = ParentUSR;
  ParentI->ChildFunctions.emplace_back(std::move(Func));
  // Info is wrapped in its parent scope so it's returned in the second position
  return {nullptr, std::unique_ptr<Info>{std::move(ParentI)}};
}

std::pair<std::unique_ptr<Info>, std::unique_ptr<Info>>
emitInfo(const EnumDecl *D, const FullComment *FC, int LineNumber,
         llvm::StringRef File, bool IsFileInRootDir, bool PublicOnly) {
  EnumInfo Enum;
  bool IsInAnonymousNamespace = false;
  populateSymbolInfo(Enum, D, FC, LineNumber, File, IsFileInRootDir,
                     IsInAnonymousNamespace);
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  Enum.Scoped = D->isScoped();
  parseEnumerators(Enum, D);

  // Put in global namespace
  if (Enum.Namespace.empty()) {
    auto ParentI = std::make_unique<NamespaceInfo>();
    ParentI->USR = SymbolID();
    ParentI->ChildEnums.emplace_back(std::move(Enum));
    ParentI->Path = getInfoRelativePath(ParentI->Namespace);
    // Info is wrapped in its parent scope so it's returned in the second
    // position
    return {nullptr, std::unique_ptr<Info>{std::move(ParentI)}};
  }

  // Wrap in enclosing scope
  switch (Enum.Namespace[0].RefType) {
  case InfoType::IT_namespace: {
    auto ParentI = std::make_unique<NamespaceInfo>();
    ParentI->USR = Enum.Namespace[0].USR;
    ParentI->ChildEnums.emplace_back(std::move(Enum));
    // Info is wrapped in its parent scope so it's returned in the second
    // position
    return {nullptr, std::unique_ptr<Info>{std::move(ParentI)}};
  }
  case InfoType::IT_record: {
    auto ParentI = std::make_unique<RecordInfo>();
    ParentI->USR = Enum.Namespace[0].USR;
    ParentI->ChildEnums.emplace_back(std::move(Enum));
    // Info is wrapped in its parent scope so it's returned in the second
    // position
    return {nullptr, std::unique_ptr<Info>{std::move(ParentI)}};
  }
  default:
    llvm_unreachable("Invalid reference type for parent namespace");
  }
}

} // namespace serialize
} // namespace doc
} // namespace clang
