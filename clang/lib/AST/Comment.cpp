//===--- Comment.cpp - Comment AST node implementation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Comment.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace comments {

const char *Comment::getCommentKindName() const {
  switch (getCommentKind()) {
  case NoCommentKind: return "NoCommentKind";
#define ABSTRACT_COMMENT(COMMENT)
#define COMMENT(CLASS, PARENT) \
  case CLASS##Kind: \
    return #CLASS;
#include "clang/AST/CommentNodes.inc"
#undef COMMENT
#undef ABSTRACT_COMMENT
  }
  llvm_unreachable("Unknown comment kind!");
}

void Comment::dump() const {
  // It is important that Comment::dump() is defined in a different TU than
  // Comment::dump(raw_ostream, SourceManager).  If both functions were defined
  // in CommentDumper.cpp, that object file would be removed by linker because
  // none of its functions are referenced by other object files, despite the
  // LLVM_ATTRIBUTE_USED.
  dump(llvm::errs(), NULL, NULL);
}

void Comment::dump(const ASTContext &Context) const {
  dump(llvm::errs(), &Context.getCommentCommandTraits(),
       &Context.getSourceManager());
}

namespace {
struct good {};
struct bad {};

template <typename T>
good implements_child_begin_end(Comment::child_iterator (T::*)() const) {
  return good();
}

static inline bad implements_child_begin_end(
                      Comment::child_iterator (Comment::*)() const) {
  return bad();
}

#define ASSERT_IMPLEMENTS_child_begin(function) \
  (void) sizeof(good(implements_child_begin_end(function)))

static inline void CheckCommentASTNodes() {
#define ABSTRACT_COMMENT(COMMENT)
#define COMMENT(CLASS, PARENT) \
  ASSERT_IMPLEMENTS_child_begin(&CLASS::child_begin); \
  ASSERT_IMPLEMENTS_child_begin(&CLASS::child_end);
#include "clang/AST/CommentNodes.inc"
#undef COMMENT
#undef ABSTRACT_COMMENT
}

#undef ASSERT_IMPLEMENTS_child_begin

} // end unnamed namespace

Comment::child_iterator Comment::child_begin() const {
  switch (getCommentKind()) {
  case NoCommentKind: llvm_unreachable("comment without a kind");
#define ABSTRACT_COMMENT(COMMENT)
#define COMMENT(CLASS, PARENT) \
  case CLASS##Kind: \
    return static_cast<const CLASS *>(this)->child_begin();
#include "clang/AST/CommentNodes.inc"
#undef COMMENT
#undef ABSTRACT_COMMENT
  }
  llvm_unreachable("Unknown comment kind!");
}

Comment::child_iterator Comment::child_end() const {
  switch (getCommentKind()) {
  case NoCommentKind: llvm_unreachable("comment without a kind");
#define ABSTRACT_COMMENT(COMMENT)
#define COMMENT(CLASS, PARENT) \
  case CLASS##Kind: \
    return static_cast<const CLASS *>(this)->child_end();
#include "clang/AST/CommentNodes.inc"
#undef COMMENT
#undef ABSTRACT_COMMENT
  }
  llvm_unreachable("Unknown comment kind!");
}

bool TextComment::isWhitespaceNoCache() const {
  for (StringRef::const_iterator I = Text.begin(), E = Text.end();
       I != E; ++I) {
    const char C = *I;
    if (C != ' ' && C != '\n' && C != '\r' &&
        C != '\t' && C != '\f' && C != '\v')
      return false;
  }
  return true;
}

bool ParagraphComment::isWhitespaceNoCache() const {
  for (child_iterator I = child_begin(), E = child_end(); I != E; ++I) {
    if (const TextComment *TC = dyn_cast<TextComment>(*I)) {
      if (!TC->isWhitespace())
        return false;
    } else
      return false;
  }
  return true;
}

const char *ParamCommandComment::getDirectionAsString(PassDirection D) {
  switch (D) {
  case ParamCommandComment::In:
    return "[in]";
  case ParamCommandComment::Out:
    return "[out]";
  case ParamCommandComment::InOut:
    return "[in,out]";
  }
  llvm_unreachable("unknown PassDirection");
}

void DeclInfo::fill() {
  assert(!IsFilled);

  // Set defaults.
  Kind = OtherKind;
  TemplateKind = NotTemplate;
  IsObjCMethod = false;
  IsInstanceMethod = false;
  IsClassMethod = false;
  ParamVars = ArrayRef<const ParmVarDecl *>();
  TemplateParameters = NULL;

  if (!CommentDecl) {
    // If there is no declaration, the defaults is our only guess.
    IsFilled = true;
    return;
  }
  CurrentDecl = CommentDecl;
  
  Decl::Kind K = CommentDecl->getKind();
  switch (K) {
  default:
    // Defaults are should be good for declarations we don't handle explicitly.
    break;
  case Decl::Function:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXDestructor:
  case Decl::CXXConversion: {
    const FunctionDecl *FD = cast<FunctionDecl>(CommentDecl);
    Kind = FunctionKind;
    ParamVars = ArrayRef<const ParmVarDecl *>(FD->param_begin(),
                                              FD->getNumParams());
    ResultType = FD->getResultType();
    unsigned NumLists = FD->getNumTemplateParameterLists();
    if (NumLists != 0) {
      TemplateKind = TemplateSpecialization;
      TemplateParameters =
          FD->getTemplateParameterList(NumLists - 1);
    }

    if (K == Decl::CXXMethod || K == Decl::CXXConstructor ||
        K == Decl::CXXDestructor || K == Decl::CXXConversion) {
      const CXXMethodDecl *MD = cast<CXXMethodDecl>(CommentDecl);
      IsInstanceMethod = MD->isInstance();
      IsClassMethod = !IsInstanceMethod;
    }
    break;
  }
  case Decl::ObjCMethod: {
    const ObjCMethodDecl *MD = cast<ObjCMethodDecl>(CommentDecl);
    Kind = FunctionKind;
    ParamVars = ArrayRef<const ParmVarDecl *>(MD->param_begin(),
                                              MD->param_size());
    ResultType = MD->getResultType();
    IsObjCMethod = true;
    IsInstanceMethod = MD->isInstanceMethod();
    IsClassMethod = !IsInstanceMethod;
    break;
  }
  case Decl::FunctionTemplate: {
    const FunctionTemplateDecl *FTD = cast<FunctionTemplateDecl>(CommentDecl);
    Kind = FunctionKind;
    TemplateKind = Template;
    const FunctionDecl *FD = FTD->getTemplatedDecl();
    ParamVars = ArrayRef<const ParmVarDecl *>(FD->param_begin(),
                                              FD->getNumParams());
    ResultType = FD->getResultType();
    TemplateParameters = FTD->getTemplateParameters();
    break;
  }
  case Decl::ClassTemplate: {
    const ClassTemplateDecl *CTD = cast<ClassTemplateDecl>(CommentDecl);
    Kind = ClassKind;
    TemplateKind = Template;
    TemplateParameters = CTD->getTemplateParameters();
    break;
  }
  case Decl::ClassTemplatePartialSpecialization: {
    const ClassTemplatePartialSpecializationDecl *CTPSD =
        cast<ClassTemplatePartialSpecializationDecl>(CommentDecl);
    Kind = ClassKind;
    TemplateKind = TemplatePartialSpecialization;
    TemplateParameters = CTPSD->getTemplateParameters();
    break;
  }
  case Decl::ClassTemplateSpecialization:
    Kind = ClassKind;
    TemplateKind = TemplateSpecialization;
    break;
  case Decl::Record:
  case Decl::CXXRecord:
    Kind = ClassKind;
    break;
  case Decl::Var:
  case Decl::Field:
  case Decl::EnumConstant:
  case Decl::ObjCIvar:
  case Decl::ObjCAtDefsField:
    Kind = VariableKind;
    break;
  case Decl::Namespace:
    Kind = NamespaceKind;
    break;
  case Decl::Typedef: {
    Kind = TypedefKind;
    // If this is a typedef to something we consider a function, extract
    // arguments and return type.
    const TypedefDecl *TD = cast<TypedefDecl>(CommentDecl);
    const TypeSourceInfo *TSI = TD->getTypeSourceInfo();
    if (!TSI)
      break;
    TypeLoc TL = TSI->getTypeLoc().getUnqualifiedLoc();
    while (true) {
      TL = TL.IgnoreParens();
      // Look through qualified types.
      if (QualifiedTypeLoc *QualifiedTL = dyn_cast<QualifiedTypeLoc>(&TL)) {
        TL = QualifiedTL->getUnqualifiedLoc();
        continue;
      }
      // Look through pointer types.
      if (PointerTypeLoc *PointerTL = dyn_cast<PointerTypeLoc>(&TL)) {
        TL = PointerTL->getPointeeLoc().getUnqualifiedLoc();
        continue;
      }
      if (BlockPointerTypeLoc *BlockPointerTL =
              dyn_cast<BlockPointerTypeLoc>(&TL)) {
        TL = BlockPointerTL->getPointeeLoc().getUnqualifiedLoc();
        continue;
      }
      if (MemberPointerTypeLoc *MemberPointerTL =
              dyn_cast<MemberPointerTypeLoc>(&TL)) {
        TL = MemberPointerTL->getPointeeLoc().getUnqualifiedLoc();
        continue;
      }
      // Is this a typedef for a function type?
      if (FunctionTypeLoc *FTL = dyn_cast<FunctionTypeLoc>(&TL)) {
        Kind = FunctionKind;
        ArrayRef<ParmVarDecl *> Params = FTL->getParams();
        ParamVars = ArrayRef<const ParmVarDecl *>(Params.data(),
                                                  Params.size());
        ResultType = FTL->getResultLoc().getType();
        break;
      }
      break;
    }
    break;
  }
  case Decl::TypeAlias:
    Kind = TypedefKind;
    break;
  case Decl::TypeAliasTemplate: {
    const TypeAliasTemplateDecl *TAT = cast<TypeAliasTemplateDecl>(CommentDecl);
    Kind = TypedefKind;
    TemplateKind = Template;
    TemplateParameters = TAT->getTemplateParameters();
    break;
  }
  case Decl::Enum:
    Kind = EnumKind;
    break;
  }

  IsFilled = true;
}

StringRef ParamCommandComment::getParamName(const FullComment *FC) const {
  assert(isParamIndexValid());
  return FC->getThisDeclInfo()->ParamVars[getParamIndex()]->getName();
}

StringRef TParamCommandComment::getParamName(const FullComment *FC) const {
  assert(isPositionValid());
  const TemplateParameterList *TPL = FC->getThisDeclInfo()->TemplateParameters;
  for (unsigned i = 0, e = getDepth(); i != e; ++i) {
    if (i == e-1)
      return TPL->getParam(getIndex(i))->getName();
    const NamedDecl *Param = TPL->getParam(getIndex(i));
    if (const TemplateTemplateParmDecl *TTP =
          dyn_cast<TemplateTemplateParmDecl>(Param))
      TPL = TTP->getTemplateParameters();
  }
  return "";
}

} // end namespace comments
} // end namespace clang

