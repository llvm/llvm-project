//===-- Serialize.cpp - ClangDoc Serializer ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Serialize.h"
#include "BitcodeWriter.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/Mangle.h"
#include "clang/Lex/Lexer.h"
#include "clang/UnifiedSymbolResolution/USRGeneration.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA1.h"

using clang::comments::FullComment;

namespace clang {
namespace doc {
namespace serialize {

namespace {
static StringRef exprToString(const clang::Expr *E) {
  clang::LangOptions Opts;
  clang::PrintingPolicy Policy(Opts);
  SmallString<16> Result;
  llvm::raw_svector_ostream OS(Result);
  E->printPretty(OS, nullptr, Policy);
  return internString(Result);
}
} // namespace

SymbolID hashUSR(llvm::StringRef USR) {
  return llvm::SHA1::hash(arrayRefFromStringRef(USR));
}

void Serializer::getTemplateParameters(
    const TemplateParameterList *TemplateParams, llvm::raw_ostream &Stream) {
  Stream << "template <";

  for (unsigned i = 0; i < TemplateParams->size(); ++i) {
    if (i > 0)
      Stream << ", ";

    const NamedDecl *Param = TemplateParams->getParam(i);
    if (const auto *TTP = llvm::dyn_cast<TemplateTypeParmDecl>(Param)) {
      if (TTP->wasDeclaredWithTypename())
        Stream << "typename";
      else
        Stream << "class";
      if (TTP->isParameterPack())
        Stream << "...";
      Stream << " " << TTP->getNameAsString();

      // We need to also handle type constraints for code like:
      //   template <class T = void>
      //   class C {};
      if (TTP->hasTypeConstraint()) {
        Stream << " = ";
        TTP->getTypeConstraint()->print(
            Stream, TTP->getASTContext().getPrintingPolicy());
      }
    } else if (const auto *NTTP =
                   llvm::dyn_cast<NonTypeTemplateParmDecl>(Param)) {
      NTTP->getType().print(Stream, NTTP->getASTContext().getPrintingPolicy());
      if (NTTP->isParameterPack())
        Stream << "...";
      Stream << " " << NTTP->getNameAsString();
    } else if (const auto *TTPD =
                   llvm::dyn_cast<TemplateTemplateParmDecl>(Param)) {
      Stream << "template <";
      getTemplateParameters(TTPD->getTemplateParameters(), Stream);
      Stream << "> class " << TTPD->getNameAsString();
    }
  }

  Stream << "> ";
}

// Extract the full function prototype from a FunctionDecl including
// Full Decl
StringRef Serializer::getFunctionPrototype(const FunctionDecl *FuncDecl) {
  llvm::SmallString<256> Result;
  llvm::raw_svector_ostream Stream(Result);
  const ASTContext &Ctx = FuncDecl->getASTContext();
  const auto *Method = llvm::dyn_cast<CXXMethodDecl>(FuncDecl);
  // If it's a templated function, handle the template parameters
  if (const auto *TmplDecl = FuncDecl->getDescribedTemplate())
    getTemplateParameters(TmplDecl->getTemplateParameters(), Stream);

  // If it's a virtual method
  if (Method && Method->isVirtual())
    Stream << "virtual ";

  // Print return type
  FuncDecl->getReturnType().print(Stream, Ctx.getPrintingPolicy());

  // Print function name
  Stream << " " << FuncDecl->getNameAsString() << "(";

  // Print parameter list with types, names, and default values
  for (unsigned I = 0; I < FuncDecl->getNumParams(); ++I) {
    if (I > 0)
      Stream << ", ";
    const ParmVarDecl *ParamDecl = FuncDecl->getParamDecl(I);
    QualType ParamType = ParamDecl->getType();
    ParamType.print(Stream, Ctx.getPrintingPolicy());

    // Print parameter name if it has one
    if (!ParamDecl->getName().empty())
      Stream << " " << ParamDecl->getNameAsString();

    // Print default argument if it exists
    if (ParamDecl->hasDefaultArg() &&
        !ParamDecl->hasUninstantiatedDefaultArg()) {
      if (const Expr *DefaultArg = ParamDecl->getDefaultArg()) {
        Stream << " = ";
        DefaultArg->printPretty(Stream, nullptr, Ctx.getPrintingPolicy());
      }
    }
  }

  // If it is a variadic function, add '...'
  if (FuncDecl->isVariadic()) {
    if (FuncDecl->getNumParams() > 0)
      Stream << ", ";
    Stream << "...";
  }

  Stream << ")";

  // If it's a const method, add 'const' qualifier
  if (Method) {
    if (Method->isDeleted())
      Stream << " = delete";
    if (Method->size_overridden_methods())
      Stream << " override";
    if (Method->hasAttr<clang::FinalAttr>())
      Stream << " final";
    if (Method->isConst())
      Stream << " const";
    if (Method->isPureVirtual())
      Stream << " = 0";
  }

  if (auto ExceptionSpecType = FuncDecl->getExceptionSpecType())
    Stream << " " << ExceptionSpecType;

  return internString(Result);
}

StringRef Serializer::getTypeAlias(const TypeAliasDecl *Alias) {
  llvm::SmallString<16> Result;
  llvm::raw_svector_ostream Stream(Result);
  const ASTContext &Ctx = Alias->getASTContext();
  if (const auto *TmplDecl = Alias->getDescribedTemplate())
    getTemplateParameters(TmplDecl->getTemplateParameters(), Stream);
  Stream << "using " << Alias->getNameAsString() << " = ";
  QualType Q = Alias->getUnderlyingType();
  Q.print(Stream, Ctx.getPrintingPolicy());

  return internString(Result);
}

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
StringRef
Serializer::getInfoRelativePath(llvm::ArrayRef<doc::Reference> Namespaces) {
  llvm::SmallString<128> Path;
  for (auto R = Namespaces.rbegin(), E = Namespaces.rend(); R != E; ++R)
    llvm::sys::path::append(Path, R->Name);
  return internString(Path);
}

StringRef Serializer::getInfoRelativePath(const Decl *D) {
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
  StringRef getCommandName(unsigned CommandID) const;
  bool isWhitespaceOnly(StringRef S) const;

  CommentInfo &CurrentCI;
};

void ClangDocCommentVisitor::parseComment(const comments::Comment *C) {
  CurrentCI.Kind = stringToCommentKind(C->getCommentKindName());
  ConstCommentVisitor<ClangDocCommentVisitor>::visit(C);

  unsigned NumChildren = C->child_count();
  if (NumChildren > 0) {
    CommentInfo *ChildrenArray =
        TransientArena.Allocate<CommentInfo>(NumChildren);
    unsigned I = 0;
    for (comments::Comment *Child :
         llvm::make_range(C->child_begin(), C->child_end())) {
      new (&ChildrenArray[I]) CommentInfo();
      ClangDocCommentVisitor Visitor(ChildrenArray[I]);
      Visitor.parseComment(Child);
      I++;
    }
    CurrentCI.Children =
        llvm::ArrayRef<CommentInfo>(ChildrenArray, NumChildren);
  }
}

void ClangDocCommentVisitor::visitTextComment(const TextComment *C) {
  if (!isWhitespaceOnly(C->getText()))
    CurrentCI.Text = C->getText().trim();
}

void ClangDocCommentVisitor::visitInlineCommandComment(
    const InlineCommandComment *C) {
  CurrentCI.Name = internString(getCommandName(C->getCommandID()));
  llvm::SmallVector<StringRef> Args;
  for (unsigned I = 0, E = C->getNumArgs(); I != E; ++I)
    Args.push_back(internString(C->getArgText(I).trim()));
  if (!Args.empty()) {
    CurrentCI.Args = allocateArray(Args, TransientArena);
  }
}

void ClangDocCommentVisitor::visitHTMLStartTagComment(
    const HTMLStartTagComment *C) {
  CurrentCI.Name = internString(C->getTagName());
  CurrentCI.SelfClosing = C->isSelfClosing();
  llvm::SmallVector<StringRef> AttrKeys;
  llvm::SmallVector<StringRef> AttrValues;
  for (unsigned I = 0, E = C->getNumAttrs(); I < E; ++I) {
    const HTMLStartTagComment::Attribute &Attr = C->getAttr(I);
    AttrKeys.push_back(internString(Attr.Name));
    AttrValues.push_back(internString(Attr.Value));
  }
  if (!AttrKeys.empty()) {
    CurrentCI.AttrKeys = allocateArray(AttrKeys, TransientArena);
  }
  if (!AttrValues.empty()) {
    CurrentCI.AttrValues = allocateArray(AttrValues, TransientArena);
  }
}

void ClangDocCommentVisitor::visitHTMLEndTagComment(
    const HTMLEndTagComment *C) {
  CurrentCI.Name = internString(C->getTagName());
  CurrentCI.SelfClosing = true;
}

void ClangDocCommentVisitor::visitBlockCommandComment(
    const BlockCommandComment *C) {
  CurrentCI.Name = internString(getCommandName(C->getCommandID()));
  llvm::SmallVector<StringRef> Args;
  for (unsigned I = 0, E = C->getNumArgs(); I < E; ++I)
    Args.push_back(internString(C->getArgText(I).trim()));
  if (!Args.empty()) {
    CurrentCI.Args = allocateArray(Args, TransientArena);
  }
}

void ClangDocCommentVisitor::visitParamCommandComment(
    const ParamCommandComment *C) {
  CurrentCI.Direction = internString(
      ParamCommandComment::getDirectionAsString(C->getDirection()));
  CurrentCI.Explicit = C->isDirectionExplicit();
  if (C->hasParamName())
    CurrentCI.ParamName = internString(C->getParamNameAsWritten());
}

void ClangDocCommentVisitor::visitTParamCommandComment(
    const TParamCommandComment *C) {
  if (C->hasParamName())
    CurrentCI.ParamName = internString(C->getParamNameAsWritten());
}

void ClangDocCommentVisitor::visitVerbatimBlockComment(
    const VerbatimBlockComment *C) {
  CurrentCI.Name = internString(getCommandName(C->getCommandID()));
  CurrentCI.CloseName = internString(C->getCloseName());
}

void ClangDocCommentVisitor::visitVerbatimBlockLineComment(
    const VerbatimBlockLineComment *C) {
  if (!isWhitespaceOnly(C->getText()))
    CurrentCI.Text = internString(C->getText());
}

void ClangDocCommentVisitor::visitVerbatimLineComment(
    const VerbatimLineComment *C) {
  if (!isWhitespaceOnly(C->getText()))
    CurrentCI.Text = internString(C->getText());
}

bool ClangDocCommentVisitor::isWhitespaceOnly(llvm::StringRef S) const {
  return llvm::all_of(S, isspace);
}

StringRef ClangDocCommentVisitor::getCommandName(unsigned CommandID) const {
  const CommandInfo *Info = CommandTraits::getBuiltinCommandInfo(CommandID);
  if (Info)
    return internString(Info->Name);
  // TODO: Add parsing for \file command.
  return "<not a builtin command>";
}

// Serializing functions.

StringRef Serializer::getSourceCode(const Decl *D, const SourceRange &R) {
  return internString(Lexer::getSourceText(
      CharSourceRange::getTokenRange(R), D->getASTContext().getSourceManager(),
      D->getASTContext().getLangOpts()));
}

template <typename T>
static std::string serialize(T &I, DiagnosticsEngine &Diags) {
  SmallString<2048> Buffer;
  llvm::BitstreamWriter Stream(Buffer);
  ClangDocBitcodeWriter Writer(Stream, Diags);
  Writer.emitBlock(I);
  return Buffer.str().str();
}

std::string serialize(OwnedPtr<Info> &I, DiagnosticsEngine &Diags) {
  switch (I->IT) {
  case InfoType::IT_namespace:
    return serialize(*static_cast<NamespaceInfo *>(getPtr(I)), Diags);
  case InfoType::IT_record:
    return serialize(*static_cast<RecordInfo *>(getPtr(I)), Diags);
  case InfoType::IT_enum:
    return serialize(*static_cast<EnumInfo *>(getPtr(I)), Diags);
  case InfoType::IT_function:
    return serialize(*static_cast<FunctionInfo *>(getPtr(I)), Diags);
  case InfoType::IT_concept:
    return serialize(*static_cast<ConceptInfo *>(getPtr(I)), Diags);
  case InfoType::IT_variable:
    return serialize(*static_cast<VarInfo *>(getPtr(I)), Diags);
  case InfoType::IT_friend:
  case InfoType::IT_typedef:
  case InfoType::IT_default:
    return "";
  }
  llvm_unreachable("unhandled enumerator");
}

void Serializer::parseFullComment(const FullComment *C, CommentInfo &CI) {
  ClangDocCommentVisitor Visitor(CI);
  Visitor.parseComment(C);
}

SymbolID Serializer::getUSRForDecl(const Decl *D) {
  llvm::SmallString<128> USR;
  if (index::generateUSRForDecl(D, USR))
    return SymbolID();
  return hashUSR(USR);
}

TagDecl *Serializer::getTagDeclForType(const QualType &T) {
  if (const TagDecl *D = T->getAsTagDecl())
    return D->getDefinition();
  return nullptr;
}

RecordDecl *Serializer::getRecordDeclForType(const QualType &T) {
  if (const RecordDecl *D = T->getAsRecordDecl())
    return D->getDefinition();
  return nullptr;
}

TypeInfo Serializer::getTypeInfoForType(const QualType &T,
                                        const PrintingPolicy &Policy) {
  const TagDecl *TD = getTagDeclForType(T);
  if (!TD) {
    TypeInfo TI = TypeInfo(Reference(SymbolID(), T.getAsString(Policy)));
    TI.IsBuiltIn = T->isBuiltinType();
    TI.IsTemplate = T->isTemplateTypeParmType();
    return TI;
  }
  InfoType IT;
  if (isa<EnumDecl>(TD)) {
    IT = InfoType::IT_enum;
  } else if (isa<RecordDecl>(TD)) {
    IT = InfoType::IT_record;
  } else {
    IT = InfoType::IT_default;
  }
  Reference R = Reference(getUSRForDecl(TD), TD->getNameAsString(), IT,
                          T.getAsString(Policy), getInfoRelativePath(TD));
  TypeInfo TI = TypeInfo(R);
  TI.IsBuiltIn = T->isBuiltinType();
  TI.IsTemplate = T->isTemplateTypeParmType();
  return TI;
}

bool Serializer::isPublic(const clang::AccessSpecifier AS,
                          const clang::Linkage Link) {
  if (AS == clang::AccessSpecifier::AS_private)
    return false;
  if ((Link == clang::Linkage::Module) || (Link == clang::Linkage::External))
    return true;
  return false; // otherwise, linkage is some form of internal linkage
}

bool Serializer::shouldSerializeInfo(bool PublicOnly,
                                     bool IsInAnonymousNamespace,
                                     const NamedDecl *D) {
  bool IsAnonymousNamespace = false;
  if (const auto *N = dyn_cast<NamespaceDecl>(D))
    IsAnonymousNamespace = N->isAnonymousNamespace();
  return !PublicOnly ||
         (!IsInAnonymousNamespace && !IsAnonymousNamespace &&
          isPublic(D->getAccessUnsafe(), D->getLinkageInternal()));
}

// The InsertChild functions insert the given info into the given scope using
// the method appropriate for that type. Some types are moved into the
// appropriate vector, while other types have Reference objects generated to
// refer to them.
//
// See MakeAndInsertIntoParent().
void Serializer::InsertChild(ScopeChildren &Scope, const NamespaceInfo &Info) {
  Scope.Namespaces.push_back(*allocateListNodeTransient<Reference>(
      Info.USR, Info.Name, InfoType::IT_namespace, Info.Name,
      getInfoRelativePath(Info.Namespace)));
}

void Serializer::InsertChild(ScopeChildren &Scope, const RecordInfo &Info) {
  Scope.Records.push_back(*allocateListNodeTransient<Reference>(
      Info.USR, Info.Name, InfoType::IT_record, Info.Name,
      getInfoRelativePath(Info.Namespace), Info.MangledName));
}

void Serializer::InsertChild(ScopeChildren &Scope, EnumInfo &Info) {
  Scope.Enums.push_back(*allocateListNodeTransient<EnumInfo>(&Info));
}

void Serializer::InsertChild(ScopeChildren &Scope, FunctionInfo &Info) {
  Scope.Functions.push_back(*allocateListNodeTransient<FunctionInfo>(&Info));
}

void Serializer::InsertChild(ScopeChildren &Scope, TypedefInfo &Info) {
  Scope.Typedefs.push_back(*allocateListNodeTransient<TypedefInfo>(&Info));
}

void Serializer::InsertChild(ScopeChildren &Scope, ConceptInfo &Info) {
  Scope.Concepts.push_back(*allocateListNodeTransient<ConceptInfo>(&Info));
}

void Serializer::InsertChild(ScopeChildren &Scope, VarInfo &Info) {
  Scope.Variables.push_back(*allocateListNodeTransient<VarInfo>(&Info));
}

// Creates a parent of the correct type for the given child and inserts it into
// that parent.
//
// This is complicated by the fact that namespaces and records are inserted by
// reference (constructing a "Reference" object with that namespace/record's
// info), while everything else is inserted by moving it directly into the child
// vectors.
//
// For namespaces and records, explicitly specify a const& template parameter
// when invoking this function:
//   MakeAndInsertIntoParent<const Record&>(...);
// Otherwise, specify an rvalue reference <EnumInfo&&> and move into the
// parameter. Since each variant is used once, it's not worth having a more
// elaborate system to automatically deduce this information.
template <typename ChildType>
OwnedPtr<Info> Serializer::makeAndInsertIntoParent(ChildType &Child) {
  if (Child.Namespace.empty()) {
    // Insert into unnamed parent namespace.
    auto ParentNS = allocatePtr<NamespaceInfo>();
    InsertChild(ParentNS->Children, Child);
    return ParentNS;
  }

  switch (Child.Namespace[0].RefType) {
  case InfoType::IT_namespace: {
    auto ParentNS = allocatePtr<NamespaceInfo>();
    ParentNS->USR = Child.Namespace[0].USR;
    InsertChild(ParentNS->Children, Child);
    return ParentNS;
  }
  case InfoType::IT_record: {
    auto ParentRec = allocatePtr<RecordInfo>();
    ParentRec->USR = Child.Namespace[0].USR;
    InsertChild(ParentRec->Children, Child);
    return ParentRec;
  }
  case InfoType::IT_default:
  case InfoType::IT_enum:
  case InfoType::IT_function:
  case InfoType::IT_typedef:
  case InfoType::IT_concept:
  case InfoType::IT_variable:
  case InfoType::IT_friend:
    break;
  }
  llvm_unreachable("Invalid reference type for parent namespace");
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
AccessSpecifier Serializer::getFinalAccessSpecifier(AccessSpecifier FirstAS,
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
void Serializer::parseFields(RecordInfo &I, const RecordDecl *D,
                             bool PublicOnly, AccessSpecifier Access) {
  SmallVector<MemberTypeInfo, 4> Members;
  for (const FieldDecl *F : D->fields()) {
    if (!shouldSerializeInfo(PublicOnly, /*IsInAnonymousNamespace=*/false, F))
      continue;
    populateMemberTypeInfo(Members, Access, F);
  }
  const auto *CxxRD = dyn_cast<CXXRecordDecl>(D);
  if (!CxxRD) {
    if (!Members.empty())
      I.Members = allocateArray<MemberTypeInfo>(Members, TransientArena);
    return;
  }
  for (Decl *CxxDecl : CxxRD->decls()) {
    auto *VD = dyn_cast<VarDecl>(CxxDecl);
    if (!VD ||
        !shouldSerializeInfo(PublicOnly, /*IsInAnonymousNamespace=*/false, VD))
      continue;

    if (VD->isStaticDataMember())
      populateMemberTypeInfo(Members, Access, VD, /*IsStatic=*/true);
  }
  if (!Members.empty())
    I.Members = allocateArray<MemberTypeInfo>(Members, TransientArena);
}

void Serializer::parseEnumerators(EnumInfo &I, const EnumDecl *D) {
  llvm::SmallVector<EnumValueInfo, 4> LocalMembers;
  for (const EnumConstantDecl *E : D->enumerators()) {
    std::string ValueExpr;
    if (const Expr *InitExpr = E->getInitExpr())
      ValueExpr = getSourceCode(D, InitExpr->getSourceRange());
    SmallString<16> ValueStr;
    E->getInitVal().toString(ValueStr);
    EnumValueInfo &Member = LocalMembers.emplace_back(
        E->getNameAsString(), ValueStr.str(), ValueExpr);
    ASTContext &Context = E->getASTContext();
    if (RawComment *Comment =
            E->getASTContext().getRawCommentForDeclNoCache(E)) {
      Comment->setAttached();
      if (comments::FullComment *Fc = Comment->parse(Context, nullptr, E)) {
        auto *NewCI = allocateListNodeTransient<CommentInfo>();
        Member.Description.push_back(*NewCI);
        parseFullComment(Fc, *NewCI->Ptr);
      }
    }
  }
  if (!LocalMembers.empty())
    I.Members = allocateArray<EnumValueInfo>(LocalMembers, TransientArena);
}

void Serializer::parseParameters(FunctionInfo &I, const FunctionDecl *D) {
  llvm::SmallVector<FieldTypeInfo, 4> LocalParams;
  auto &LO = D->getLangOpts();
  for (const ParmVarDecl *P : D->parameters()) {
    FieldTypeInfo &FieldInfo = LocalParams.emplace_back(
        getTypeInfoForType(P->getOriginalType(), LO), P->getNameAsString());
    if (std::optional<StringRef> DefaultValue =
            getSourceCode(D, P->getDefaultArgRange()))
      FieldInfo.DefaultValue = *DefaultValue;
  }
  if (!LocalParams.empty())
    I.Params = allocateArray<FieldTypeInfo>(LocalParams, TransientArena);
}

// TODO: Remove the serialization of Parents and VirtualParents, this
// information is also extracted in the other definition of parseBases.
void Serializer::parseBases(RecordInfo &I, const CXXRecordDecl *D) {
  // Don't parse bases if this isn't a definition.
  if (!D->isThisDeclarationADefinition())
    return;

  llvm::SmallVector<Reference, 4> LocalParents;
  for (const CXXBaseSpecifier &B : D->bases()) {
    if (B.isVirtual())
      continue;
    if (const auto *Ty = B.getType()->getAs<TemplateSpecializationType>()) {
      const TemplateDecl *D = Ty->getTemplateName().getAsTemplateDecl();
      LocalParents.emplace_back(getUSRForDecl(D), B.getType().getAsString(),
                                InfoType::IT_record, B.getType().getAsString());
    } else if (const RecordDecl *P = getRecordDeclForType(B.getType()))
      LocalParents.emplace_back(
          getUSRForDecl(P), P->getNameAsString(), InfoType::IT_record,
          P->getQualifiedNameAsString(), internString(getInfoRelativePath(P)));
    else
      LocalParents.emplace_back(SymbolID(), B.getType().getAsString());
  }
  if (!LocalParents.empty())
    I.Parents = allocateArray<Reference>(LocalParents, TransientArena);

  llvm::SmallVector<Reference, 4> LocalVirtualParents;
  for (const CXXBaseSpecifier &B : D->vbases()) {
    if (const RecordDecl *P = getRecordDeclForType(B.getType()))
      LocalVirtualParents.emplace_back(
          getUSRForDecl(P), P->getNameAsString(), InfoType::IT_record,
          P->getQualifiedNameAsString(), internString(getInfoRelativePath(P)));
    else
      LocalVirtualParents.emplace_back(SymbolID(), B.getType().getAsString());
  }
  if (!LocalVirtualParents.empty())
    I.VirtualParents =
        allocateArray<Reference>(LocalVirtualParents, TransientArena);
}

template <typename T>
void Serializer::populateParentNamespaces(
    llvm::SmallVector<Reference, 4> &Namespaces, const T *D,
    bool &IsInAnonymousNamespace) {
  const DeclContext *DC = D->getDeclContext();
  do {
    if (const auto *N = dyn_cast<NamespaceDecl>(DC)) {
      std::string Namespace;
      if (N->isAnonymousNamespace()) {
        Namespace = "@nonymous_namespace";
        IsInAnonymousNamespace = true;
      } else
        Namespace = N->getNameAsString();
      Namespaces.emplace_back(getUSRForDecl(N), Namespace,
                              InfoType::IT_namespace,
                              N->getQualifiedNameAsString());
    } else if (const auto *N = dyn_cast<RecordDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_record,
                              N->getQualifiedNameAsString());
    else if (const auto *N = dyn_cast<FunctionDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_function,
                              N->getQualifiedNameAsString());
    else if (const auto *N = dyn_cast<EnumDecl>(DC))
      Namespaces.emplace_back(getUSRForDecl(N), N->getNameAsString(),
                              InfoType::IT_enum, N->getQualifiedNameAsString());
  } while ((DC = DC->getParent()));
  // The global namespace should be added to the list of namespaces if the decl
  // corresponds to a Record and if it doesn't have any namespace (because this
  // means it's in the global namespace). Also if its outermost namespace is a
  // record because that record matches the previous condition mentioned.
  if ((Namespaces.empty() && isa<RecordDecl>(D)) ||
      (!Namespaces.empty() && Namespaces.back().RefType == InfoType::IT_record))
    Namespaces.emplace_back(SymbolID(), "GlobalNamespace",
                            InfoType::IT_namespace);
}

void Serializer::populateTemplateParameters(
    std::optional<TemplateInfo> &TemplateInfo, const clang::Decl *D) {
  if (const TemplateParameterList *ParamList =
          D->getDescribedTemplateParams()) {
    if (!TemplateInfo) {
      TemplateInfo.emplace();
    }
    llvm::SmallVector<TemplateParamInfo, 4> LocalParams;
    for (const NamedDecl *ND : *ParamList) {
      LocalParams.emplace_back(getSourceCode(ND, ND->getSourceRange()));
    }
    if (!LocalParams.empty())
      TemplateInfo->Params =
          allocateArray<TemplateParamInfo>(LocalParams, TransientArena);
  }
}

TemplateParamInfo
Serializer::convertTemplateArgToInfo(const clang::Decl *D,
                                     const TemplateArgument &Arg) {
  // The TemplateArgument's pretty printing handles all the normal cases
  // well enough for our requirements.
  std::string Str;
  llvm::raw_string_ostream Stream(Str);
  Arg.print(PrintingPolicy(D->getLangOpts()), Stream, false);
  return TemplateParamInfo(Str);
}

// Check if the DeclKind is one for which we support contextual relationships.
// There might be other ContextDecls, like blocks, that we currently don't
// handle at all.
bool Serializer::isSupportedContext(Decl::Kind DeclKind) {
  switch (DeclKind) {
  case Decl::Kind::Record:
  case Decl::Kind::CXXRecord:
  case Decl::Kind::ClassTemplateSpecialization:
  case Decl::Kind::ClassTemplatePartialSpecialization:
  case Decl::Kind::Namespace:
    return true;
  default:
    return false;
  }
}

void Serializer::findParent(Info &I, const Decl *D) {
  assert(D && "Invalid Decl");

  // Only walk up contexts if D is a record or namespace.
  if (!isSupportedContext(D->getKind()))
    return;

  const DeclContext *ParentCtx = dyn_cast<DeclContext>(D)->getLexicalParent();
  while (ParentCtx) {
    if (isSupportedContext(ParentCtx->getDeclKind())) {
      // Break when we reach the first record or namespace.
      I.ParentUSR = getUSRForDecl(dyn_cast<Decl>(ParentCtx));
      break;
    }
    ParentCtx = ParentCtx->getParent();
  }
}

template <typename T>
void Serializer::populateInfo(Info &I, const T *D, const FullComment *C,
                              bool &IsInAnonymousNamespace) {
  I.USR = getUSRForDecl(D);
  findParent(I, D);

  if (auto ConversionDecl = dyn_cast_or_null<CXXConversionDecl>(D);
      ConversionDecl && ConversionDecl->getConversionType()
                            .getTypePtr()
                            ->isTemplateTypeParmType())
    I.Name = internString("operator " +
                          ConversionDecl->getConversionType().getAsString());
  else
    I.Name = internString(D->getNameAsString());
  llvm::SmallVector<Reference, 4> LocalNamespaces;
  populateParentNamespaces(LocalNamespaces, D, IsInAnonymousNamespace);
  if (!LocalNamespaces.empty())
    I.Namespace = allocateArray<Reference>(LocalNamespaces, TransientArena);
  if (C) {

    auto *NewCI = allocateListNodeTransient<CommentInfo>();
    I.Description.push_back(*NewCI);
    parseFullComment(C, *NewCI->Ptr);
  }
}

template <typename T>
void Serializer::populateSymbolInfo(SymbolInfo &I, const T *D,
                                    const FullComment *C, Location Loc,
                                    bool &IsInAnonymousNamespace) {
  populateInfo(I, D, C, IsInAnonymousNamespace);
  if (D->isThisDeclarationADefinition())
    I.DefLoc = Loc;
  else {
    I.Loc.push_back(*allocateListNodeTransient<Location>(Loc));
  }

  auto *Mangler = ItaniumMangleContext::create(
      D->getASTContext(), D->getASTContext().getDiagnostics());
  std::string MangledName;
  llvm::raw_string_ostream MangledStream(MangledName);
  if (auto *CXXD = dyn_cast<CXXRecordDecl>(D))
    Mangler->mangleCXXVTable(CXXD, MangledStream);
  else
    MangledStream << D->getNameAsString();
  // A 250 length limit was chosen since 255 is a common limit across
  // different filesystems, with a 5 character buffer for file extensions.
  if (MangledName.size() > 250) {
    auto SymbolID = llvm::toStringRef(llvm::toHex(I.USR)).str();
    I.MangledName =
        internString(MangledName.substr(0, 250 - SymbolID.size()) + SymbolID);
  } else
    I.MangledName = internString(MangledName);
  delete Mangler;
}

void Serializer::handleCompoundConstraints(
    const Expr *Constraint,
    llvm::SmallVectorImpl<ConstraintInfo> &ConstraintInfos) {
  if (Constraint->getStmtClass() == Stmt::ParenExprClass) {
    handleCompoundConstraints(dyn_cast<ParenExpr>(Constraint)->getSubExpr(),
                              ConstraintInfos);
  } else if (Constraint->getStmtClass() == Stmt::BinaryOperatorClass) {
    auto *BinaryOpExpr = dyn_cast<BinaryOperator>(Constraint);
    handleCompoundConstraints(BinaryOpExpr->getLHS(), ConstraintInfos);
    handleCompoundConstraints(BinaryOpExpr->getRHS(), ConstraintInfos);
  } else if (Constraint->getStmtClass() ==
             Stmt::ConceptSpecializationExprClass) {
    auto *Concept = dyn_cast<ConceptSpecializationExpr>(Constraint);
    ConstraintInfo CI(getUSRForDecl(Concept->getNamedConcept()),
                      Concept->getNamedConcept()->getNameAsString());
    CI.ConstraintExpr = internString(exprToString(Concept));
    ConstraintInfos.push_back(CI);
  }
}

void Serializer::populateConstraints(TemplateInfo &I, const TemplateDecl *D) {
  if (!D || !D->hasAssociatedConstraints())
    return;

  SmallVector<AssociatedConstraint> AssociatedConstraints;
  D->getAssociatedConstraints(AssociatedConstraints);
  SmallVector<ConstraintInfo, 4> LocalConstraints;
  for (const auto &Constraint : AssociatedConstraints) {
    if (!Constraint)
      continue;

    // TODO: Investigate if atomic constraints need to be handled specifically.
    if (const auto *ConstraintExpr =
            dyn_cast_or_null<ConceptSpecializationExpr>(
                Constraint.ConstraintExpr)) {
      ConstraintInfo CI(getUSRForDecl(ConstraintExpr->getNamedConcept()),
                        ConstraintExpr->getNamedConcept()->getNameAsString());
      CI.ConstraintExpr = internString(exprToString(ConstraintExpr));
      LocalConstraints.push_back(std::move(CI));
    } else {
      handleCompoundConstraints(Constraint.ConstraintExpr, LocalConstraints);
    }
  }
  if (!LocalConstraints.empty())
    I.Constraints =
        allocateArray<ConstraintInfo>(LocalConstraints, TransientArena);
}

void Serializer::populateFunctionInfo(FunctionInfo &I, const FunctionDecl *D,
                                      const FullComment *FC, Location Loc,
                                      bool &IsInAnonymousNamespace) {
  populateSymbolInfo(I, D, FC, Loc, IsInAnonymousNamespace);
  auto &LO = D->getLangOpts();
  I.ReturnType = getTypeInfoForType(D->getReturnType(), LO);
  I.Prototype = getFunctionPrototype(D);
  parseParameters(I, D);
  I.IsStatic = D->isStatic();

  populateTemplateParameters(I.Template, D);
  if (I.Template)
    populateConstraints(I.Template.value(), D->getDescribedFunctionTemplate());

  // Handle function template specializations.
  if (const FunctionTemplateSpecializationInfo *FTSI =
          D->getTemplateSpecializationInfo()) {
    if (!I.Template)
      I.Template.emplace();
    I.Template->Specialization.emplace();
    auto &Specialization = *I.Template->Specialization;

    Specialization.SpecializationOf = getUSRForDecl(FTSI->getTemplate());

    // Template parameters to the specialization.
    if (FTSI->TemplateArguments) {
      SmallVector<TemplateParamInfo, 4> LocalParams;
      for (const TemplateArgument &Arg : FTSI->TemplateArguments->asArray()) {
        LocalParams.push_back(convertTemplateArgToInfo(D, Arg));
      }
      if (!LocalParams.empty())
        Specialization.Params =
            allocateArray<TemplateParamInfo>(LocalParams, TransientArena);
    }
  }
}

// TODO: Rename this, since this doesn't populate anything besides comments and
// isn't exclusive to members
template <typename T>
void Serializer::populateMemberTypeInfo(T &I, const Decl *D) {
  assert(D && "Expect non-null FieldDecl in populateMemberTypeInfo");

  ASTContext &Context = D->getASTContext();
  // TODO investigate whether we can use ASTContext::getCommentForDecl instead
  // of this logic. See also similar code in Mapper.cpp.
  RawComment *Comment = Context.getRawCommentForDeclNoCache(D);
  if (!Comment)
    return;

  Comment->setAttached();
  if (comments::FullComment *Fc = Comment->parse(Context, nullptr, D)) {
    auto *NewCI = allocateListNodeTransient<CommentInfo>();
    I.Description.push_back(*NewCI);
    parseFullComment(Fc, *NewCI->Ptr);
  }
}

void Serializer::populateMemberTypeInfo(
    SmallVectorImpl<MemberTypeInfo> &Members, AccessSpecifier &Access,
    const DeclaratorDecl *D, bool IsStatic) {
  // Use getAccessUnsafe so that we just get the default AS_none if it's not
  // valid, as opposed to an assert.
  MemberTypeInfo &NewMember = Members.emplace_back(
      getTypeInfoForType(D->getTypeSourceInfo()->getType(), D->getLangOpts()),
      D->getNameAsString(),
      getFinalAccessSpecifier(Access, D->getAccessUnsafe()), IsStatic);
  populateMemberTypeInfo(NewMember, D);
}

void Serializer::parseBases(llvm::SmallVectorImpl<BaseRecordInfo> &Bases,
                            const CXXRecordDecl *D, bool IsFileInRootDir,
                            bool PublicOnly, bool IsParent,
                            AccessSpecifier ParentAccess) {
  // Don't parse bases if this isn't a definition.
  if (!D->isThisDeclarationADefinition())
    return;
  for (const CXXBaseSpecifier &B : D->bases()) {
    if (const auto *Base = B.getType()->getAsCXXRecordDecl()) {
      if (Base->isCompleteDefinition()) {
        // Initialized without USR and name, this will be set in the following
        // if-else stmt.
        BaseRecordInfo BI(
            {}, "", internString(getInfoRelativePath(Base)), B.isVirtual(),
            getFinalAccessSpecifier(ParentAccess, B.getAccessSpecifier()),
            IsParent);
        if (const auto *Ty = B.getType()->getAs<TemplateSpecializationType>()) {
          const TemplateDecl *D = Ty->getTemplateName().getAsTemplateDecl();
          BI.USR = getUSRForDecl(D);
          BI.Name = internString(B.getType().getAsString());
        } else {
          BI.USR = getUSRForDecl(Base);
          BI.Name = internString(Base->getNameAsString());
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
            FI.IsStatic = MD->isStatic();
            // The seventh arg in populateFunctionInfo is a boolean passed by
            // reference, its value is not relevant in here so it's not used
            // anywhere besides the function call.
            bool IsInAnonymousNamespace;
            populateFunctionInfo(FI, MD, /*FullComment=*/{}, /*Location=*/{},
                                 IsInAnonymousNamespace);
            FI.Access =
                getFinalAccessSpecifier(BI.Access, MD->getAccessUnsafe());
            FunctionInfo *FIPtr = allocatePtr<FunctionInfo>(std::move(FI));
            BI.Children.Functions.push_back(
                *allocatePtr<InfoNode<FunctionInfo>>(FIPtr));
          }
        Bases.emplace_back(std::move(BI));
        // Call this function recursively to get the inherited classes of
        // this base; these new bases will also get stored in the original
        // RecordInfo: I.
        parseBases(Bases, Base, IsFileInRootDir, PublicOnly, false,
                   Bases.back().Access);
      }
    }
  }
}

std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
Serializer::emitInfo(const NamespaceDecl *D, const FullComment *FC,
                     Location Loc, bool PublicOnly) {
  auto NSI = allocatePtr<NamespaceInfo>();
  bool IsInAnonymousNamespace = false;
  populateInfo(*NSI, D, FC, IsInAnonymousNamespace);
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  NSI->Name = D->isAnonymousNamespace() ? "@nonymous_namespace" : NSI->Name;
  NSI->Path = getInfoRelativePath(NSI->Namespace);
  if (NSI->Namespace.empty() && NSI->USR == SymbolID())
    return {OwnedPtr<Info>{std::move(NSI)}, nullptr};

  // Namespaces are inserted into the parent by reference, so we need to return
  // both the parent and the record itself.
  return {std::move(NSI), makeAndInsertIntoParent<const NamespaceInfo &>(*NSI)};
}

void Serializer::parseFriends(RecordInfo &RI, const CXXRecordDecl *D) {
  if (!D->hasDefinition() || !D->hasFriends())
    return;

  llvm::SmallVector<FriendInfo, 4> LocalFriends;

  for (const FriendDecl *FD : D->friends()) {
    if (FD->isUnsupportedFriend())
      continue;

    FriendInfo F(InfoType::IT_friend, getUSRForDecl(FD));
    const auto *ActualDecl = FD->getFriendDecl();
    if (!ActualDecl) {
      const auto *FriendTypeInfo = FD->getFriendType();
      if (!FriendTypeInfo)
        continue;
      ActualDecl = FriendTypeInfo->getType()->getAsCXXRecordDecl();

      if (!ActualDecl)
        continue;
      F.IsClass = true;
    }

    if (const auto *ActualTD = dyn_cast_or_null<TemplateDecl>(ActualDecl)) {
      if (isa<RecordDecl>(ActualTD->getTemplatedDecl()))
        F.IsClass = true;
      F.Template.emplace();
      llvm::SmallVector<TemplateParamInfo, 4> LocalParams;
      for (const auto *Param : ActualTD->getTemplateParameters()->asArray())
        LocalParams.emplace_back(getSourceCode(Param, Param->getSourceRange()));
      if (!LocalParams.empty())
        F.Template->Params =
            allocateArray<TemplateParamInfo>(LocalParams, TransientArena);
      ActualDecl = ActualTD->getTemplatedDecl();
    }

    if (auto *FuncDecl = dyn_cast_or_null<FunctionDecl>(ActualDecl)) {
      FunctionInfo TempInfo;
      parseParameters(TempInfo, FuncDecl);
      F.Params = allocateArray<FieldTypeInfo>(TempInfo.Params, TransientArena);
      F.ReturnType = getTypeInfoForType(FuncDecl->getReturnType(),
                                        FuncDecl->getLangOpts());
    }

    F.Ref =
        Reference(getUSRForDecl(ActualDecl), ActualDecl->getNameAsString(),
                  InfoType::IT_default, ActualDecl->getQualifiedNameAsString(),
                  getInfoRelativePath(ActualDecl));

    populateMemberTypeInfo(F, ActualDecl);
    LocalFriends.push_back(std::move(F));
  }
  if (!LocalFriends.empty())
    RI.Friends = allocateArray<FriendInfo>(LocalFriends, TransientArena);
}

std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
Serializer::emitInfo(const RecordDecl *D, const FullComment *FC, Location Loc,
                     bool PublicOnly) {

  auto RI = allocatePtr<RecordInfo>();
  bool IsInAnonymousNamespace = false;

  populateSymbolInfo(*RI, D, FC, Loc, IsInAnonymousNamespace);
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  RI->TagType = D->getTagKind();
  parseFields(*RI, D, PublicOnly);

  if (const auto *C = dyn_cast<CXXRecordDecl>(D)) {
    if (const TypedefNameDecl *TD = C->getTypedefNameForAnonDecl()) {
      RI->Name = internString(TD->getNameAsString());
      RI->IsTypeDef = true;
    }
    // TODO: remove first call to parseBases, that function should be deleted
    parseBases(*RI, C);
    llvm::SmallVector<BaseRecordInfo, 4> LocalBases;
    parseBases(LocalBases, C, /*IsFileInRootDir=*/true, PublicOnly,
               /*IsParent=*/true);
    if (!LocalBases.empty())
      RI->Bases = allocateArray<BaseRecordInfo>(LocalBases, TransientArena);
    parseFriends(*RI, C);
  }
  RI->Path = internString(getInfoRelativePath(RI->Namespace));

  populateTemplateParameters(RI->Template, D);
  if (RI->Template)
    populateConstraints(RI->Template.value(), D->getDescribedTemplate());

  // Full and partial specializations.
  if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    if (!RI->Template)
      RI->Template.emplace();
    RI->Template->Specialization.emplace();
    auto &Specialization = *RI->Template->Specialization;

    // What this is a specialization of.
    auto SpecOf = CTSD->getSpecializedTemplateOrPartial();
    if (auto *SpecTD = dyn_cast<ClassTemplateDecl *>(SpecOf))
      Specialization.SpecializationOf = getUSRForDecl(SpecTD);
    else if (auto *SpecTD =
                 dyn_cast<ClassTemplatePartialSpecializationDecl *>(SpecOf))
      Specialization.SpecializationOf = getUSRForDecl(SpecTD);

    // Parameters to the specialization. For partial specializations, get the
    // parameters "as written" from the ClassTemplatePartialSpecializationDecl
    // because the non-explicit template parameters will have generated internal
    // placeholder names rather than the names the user typed that match the
    // template parameters.
    if (const ClassTemplatePartialSpecializationDecl *CTPSD =
            dyn_cast<ClassTemplatePartialSpecializationDecl>(D)) {
      if (const ASTTemplateArgumentListInfo *AsWritten =
              CTPSD->getTemplateArgsAsWritten()) {
        llvm::SmallVector<TemplateParamInfo, 4> LocalParams;
        for (unsigned Idx = 0; Idx < AsWritten->getNumTemplateArgs(); Idx++) {
          LocalParams.emplace_back(
              getSourceCode(D, (*AsWritten)[Idx].getSourceRange()));
        }
        if (!LocalParams.empty())
          Specialization.Params =
              allocateArray<TemplateParamInfo>(LocalParams, TransientArena);
      }
    } else {
      llvm::SmallVector<TemplateParamInfo, 4> LocalParams;
      for (const TemplateArgument &Arg : CTSD->getTemplateArgs().asArray()) {
        LocalParams.push_back(convertTemplateArgToInfo(D, Arg));
      }
      if (!LocalParams.empty())
        Specialization.Params =
            allocateArray<TemplateParamInfo>(LocalParams, TransientArena);
    }
  }

  // Records are inserted into the parent by reference, so we need to return
  // both the parent and the record itself.
  auto Parent = makeAndInsertIntoParent<const RecordInfo &>(*RI);
  return {std::move(RI), std::move(Parent)};
}

std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
Serializer::emitInfo(const FunctionDecl *D, const FullComment *FC, Location Loc,
                     bool PublicOnly) {
  FunctionInfo *Func = allocatePtr<FunctionInfo>();
  bool IsInAnonymousNamespace = false;
  populateFunctionInfo(*Func, D, FC, Loc, IsInAnonymousNamespace);
  Func->Access = clang::AccessSpecifier::AS_none;
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  // Info is wrapped in its parent scope so is returned in the second position.
  return {nullptr, makeAndInsertIntoParent(*Func)};
}

std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
Serializer::emitInfo(const CXXMethodDecl *D, const FullComment *FC,
                     Location Loc, bool PublicOnly) {
  FunctionInfo *Func = allocatePtr<FunctionInfo>();
  bool IsInAnonymousNamespace = false;
  populateFunctionInfo(*Func, D, FC, Loc, IsInAnonymousNamespace);
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  Func->IsMethod = true;
  Func->IsStatic = D->isStatic();

  const NamedDecl *Parent = nullptr;
  if (const auto *SD =
          dyn_cast<ClassTemplateSpecializationDecl>(D->getParent()))
    Parent = SD->getSpecializedTemplate();
  else
    Parent = D->getParent();

  SymbolID ParentUSR = getUSRForDecl(Parent);
  Func->Parent =
      Reference{ParentUSR, Parent->getNameAsString(), InfoType::IT_record,
                Parent->getQualifiedNameAsString()};
  Func->Access = D->getAccess();

  // Info is wrapped in its parent scope so is returned in the second position.
  return {nullptr, makeAndInsertIntoParent(*Func)};
}

void Serializer::extractCommentFromDecl(const Decl *D, TypedefInfo &Info) {
  assert(D && "Invalid Decl when extracting comment");
  ASTContext &Context = D->getASTContext();
  RawComment *Comment = Context.getRawCommentForDeclNoCache(D);
  if (!Comment)
    return;

  Comment->setAttached();
  if (comments::FullComment *Fc = Comment->parse(Context, nullptr, D)) {
    auto *NewCI = allocateListNodeTransient<CommentInfo>();
    Info.Description.push_back(*NewCI);
    parseFullComment(Fc, *NewCI->Ptr);
  }
}

std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
Serializer::emitInfo(const TypedefDecl *D, const FullComment *FC, Location Loc,
                     bool PublicOnly) {
  TypedefInfo *Info = allocatePtr<TypedefInfo>();
  bool IsInAnonymousNamespace = false;
  populateInfo(*Info, D, FC, IsInAnonymousNamespace);

  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  Info->DefLoc = Loc;
  auto &LO = D->getLangOpts();
  Info->Underlying = getTypeInfoForType(D->getUnderlyingType(), LO);
  populateTemplateParameters(Info->Template, D);
  if (Info->Template)
    populateConstraints(Info->Template.value(), D->getDescribedTemplate());

  if (Info->Underlying.Type.Name.empty()) {
    // Typedef for an unnamed type. This is like "typedef struct { } Foo;"
    // The record serializer explicitly checks for this syntax and constructs
    // a record with that name, so we don't want to emit a duplicate here.
    return {};
  }
  Info->IsUsing = false;
  extractCommentFromDecl(D, *Info);

  // Info is wrapped in its parent scope so is returned in the second position.
  return {nullptr, makeAndInsertIntoParent(*Info)};
}

// A type alias is a C++ "using" declaration for a type. It gets mapped to a
// TypedefInfo with the IsUsing flag set.
std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
Serializer::emitInfo(const TypeAliasDecl *D, const FullComment *FC,
                     Location Loc, bool PublicOnly) {
  TypedefInfo *Info = allocatePtr<TypedefInfo>();
  bool IsInAnonymousNamespace = false;
  populateInfo(*Info, D, FC, IsInAnonymousNamespace);
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  Info->DefLoc = Loc;
  const LangOptions &LO = D->getLangOpts();
  Info->Underlying = getTypeInfoForType(D->getUnderlyingType(), LO);
  Info->TypeDeclaration = getTypeAlias(D);
  Info->IsUsing = true;
  populateTemplateParameters(Info->Template, D);
  if (Info->Template)
    populateConstraints(Info->Template.value(), D->getDescribedAliasTemplate());

  extractCommentFromDecl(D, *Info);

  // Info is wrapped in its parent scope so is returned in the second position.
  return {nullptr, makeAndInsertIntoParent(*Info)};
}

std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
Serializer::emitInfo(const EnumDecl *D, const FullComment *FC, Location Loc,
                     bool PublicOnly) {
  EnumInfo *Enum = allocatePtr<EnumInfo>();
  bool IsInAnonymousNamespace = false;
  populateSymbolInfo(*Enum, D, FC, Loc, IsInAnonymousNamespace);

  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  Enum->Scoped = D->isScoped();
  if (const TypeSourceInfo *TSI = D->getIntegerTypeSourceInfo()) {
    auto Name = TSI->getType().getAsString();
    Enum->BaseType = TypeInfo(Name, Name);
  }
  parseEnumerators(*Enum, D);

  // Info is wrapped in its parent scope so is returned in the second position.
  return {nullptr, makeAndInsertIntoParent(*Enum)};
}

std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
Serializer::emitInfo(const ConceptDecl *D, const FullComment *FC,
                     const Location &Loc, bool PublicOnly) {
  ConceptInfo *Concept = allocatePtr<ConceptInfo>();

  bool IsInAnonymousNamespace = false;
  populateInfo(*Concept, D, FC, IsInAnonymousNamespace);
  Concept->IsType = D->isTypeConcept();
  Concept->DefLoc = Loc;
  Concept->ConstraintExpression = exprToString(D->getConstraintExpr());

  if (auto *ConceptParams = D->getTemplateParameters()) {
    llvm::SmallVector<TemplateParamInfo, 4> LocalParams;
    for (const auto *Param : ConceptParams->asArray()) {
      LocalParams.emplace_back(getSourceCode(Param, Param->getSourceRange()));
    }
    if (!LocalParams.empty())
      Concept->Template.Params =
          allocateArray<TemplateParamInfo>(LocalParams, TransientArena);
  }

  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  return {nullptr, makeAndInsertIntoParent(*Concept)};
}

std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
Serializer::emitInfo(const VarDecl *D, const FullComment *FC,
                     const Location &Loc, bool PublicOnly) {
  VarInfo *Var = allocatePtr<VarInfo>();
  bool IsInAnonymousNamespace = false;
  populateSymbolInfo(*Var, D, FC, Loc, IsInAnonymousNamespace);
  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  if (D->getStorageClass() == StorageClass::SC_Static)
    Var->IsStatic = true;
  Var->Type =
      getTypeInfoForType(D->getType(), D->getASTContext().getPrintingPolicy());

  if (!shouldSerializeInfo(PublicOnly, IsInAnonymousNamespace, D))
    return {};

  return {nullptr, makeAndInsertIntoParent(*Var)};
}

} // namespace serialize
} // namespace doc
} // namespace clang
