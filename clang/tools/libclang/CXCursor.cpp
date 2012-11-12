//===- CXCursor.cpp - Routines for manipulating CXCursors -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXCursors. It should be the
// only file that has internal knowledge of the encoding of the data in
// CXCursor.
//
//===----------------------------------------------------------------------===//

#include "CXTranslationUnit.h"
#include "CXCursor.h"
#include "CXType.h"
#include "CXString.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang-c/Index.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace cxcursor;

CXCursor cxcursor::MakeCXCursorInvalid(CXCursorKind K, CXTranslationUnit TU) {
  assert(K >= CXCursor_FirstInvalid && K <= CXCursor_LastInvalid);
  CXCursor C = { K, 0, { 0, 0, TU } };
  return C;
}

static CXCursorKind GetCursorKind(const Attr *A) {
  assert(A && "Invalid arguments!");
  switch (A->getKind()) {
    default: break;
    case attr::IBAction: return CXCursor_IBActionAttr;
    case attr::IBOutlet: return CXCursor_IBOutletAttr;
    case attr::IBOutletCollection: return CXCursor_IBOutletCollectionAttr;
    case attr::Final: return CXCursor_CXXFinalAttr;
    case attr::Override: return CXCursor_CXXOverrideAttr;
    case attr::Annotate: return CXCursor_AnnotateAttr;
    case attr::AsmLabel: return CXCursor_AsmLabelAttr;
  }

  return CXCursor_UnexposedAttr;
}

CXCursor cxcursor::MakeCXCursor(const Attr *A, Decl *Parent,
                                CXTranslationUnit TU) {
  assert(A && Parent && TU && "Invalid arguments!");
  CXCursor C = { GetCursorKind(A), 0, { Parent, (void*)A, TU } };
  return C;
}

CXCursor cxcursor::MakeCXCursor(Decl *D, CXTranslationUnit TU,
                                SourceRange RegionOfInterest,
                                bool FirstInDeclGroup) {
  assert(D && TU && "Invalid arguments!");

  CXCursorKind K = getCursorKindForDecl(D);

  if (K == CXCursor_ObjCClassMethodDecl ||
      K == CXCursor_ObjCInstanceMethodDecl) {
    int SelectorIdIndex = -1;
    // Check if cursor points to a selector id.
    if (RegionOfInterest.isValid() &&
        RegionOfInterest.getBegin() == RegionOfInterest.getEnd()) {
      SmallVector<SourceLocation, 16> SelLocs;
      cast<ObjCMethodDecl>(D)->getSelectorLocs(SelLocs);
      SmallVector<SourceLocation, 16>::iterator
        I=std::find(SelLocs.begin(), SelLocs.end(),RegionOfInterest.getBegin());
      if (I != SelLocs.end())
        SelectorIdIndex = I - SelLocs.begin();
    }
    CXCursor C = { K, SelectorIdIndex,
                   { D, (void*)(intptr_t) (FirstInDeclGroup ? 1 : 0), TU }};
    return C;
  }
  
  CXCursor C = { K, 0, { D, (void*)(intptr_t) (FirstInDeclGroup ? 1 : 0), TU }};
  return C;
}

CXCursor cxcursor::MakeCXCursor(Stmt *S, Decl *Parent, CXTranslationUnit TU,
                                SourceRange RegionOfInterest) {
  assert(S && TU && "Invalid arguments!");
  CXCursorKind K = CXCursor_NotImplemented;
  
  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass:
    break;
  
  case Stmt::CaseStmtClass:
    K = CXCursor_CaseStmt;
    break;
  
  case Stmt::DefaultStmtClass:
    K = CXCursor_DefaultStmt;
    break;
  
  case Stmt::IfStmtClass:
    K = CXCursor_IfStmt;
    break;
  
  case Stmt::SwitchStmtClass:
    K = CXCursor_SwitchStmt;
    break;
  
  case Stmt::WhileStmtClass:
    K = CXCursor_WhileStmt;
    break;
  
  case Stmt::DoStmtClass:
    K = CXCursor_DoStmt;
    break;
  
  case Stmt::ForStmtClass:
    K = CXCursor_ForStmt;
    break;
  
  case Stmt::GotoStmtClass:
    K = CXCursor_GotoStmt;
    break;
  
  case Stmt::IndirectGotoStmtClass:
    K = CXCursor_IndirectGotoStmt;
    break;
  
  case Stmt::ContinueStmtClass:
    K = CXCursor_ContinueStmt;
    break;
  
  case Stmt::BreakStmtClass:
    K = CXCursor_BreakStmt;
    break;
  
  case Stmt::ReturnStmtClass:
    K = CXCursor_ReturnStmt;
    break;
  
  case Stmt::GCCAsmStmtClass:
    K = CXCursor_GCCAsmStmt;
    break;

  case Stmt::MSAsmStmtClass:
    K = CXCursor_MSAsmStmt;
    break;
  
  case Stmt::ObjCAtTryStmtClass:
    K = CXCursor_ObjCAtTryStmt;
    break;
  
  case Stmt::ObjCAtCatchStmtClass:
    K = CXCursor_ObjCAtCatchStmt;
    break;
  
  case Stmt::ObjCAtFinallyStmtClass:
    K = CXCursor_ObjCAtFinallyStmt;
    break;
  
  case Stmt::ObjCAtThrowStmtClass:
    K = CXCursor_ObjCAtThrowStmt;
    break;
  
  case Stmt::ObjCAtSynchronizedStmtClass:
    K = CXCursor_ObjCAtSynchronizedStmt;
    break;
  
  case Stmt::ObjCAutoreleasePoolStmtClass:
    K = CXCursor_ObjCAutoreleasePoolStmt;
    break;
  
  case Stmt::ObjCForCollectionStmtClass:
    K = CXCursor_ObjCForCollectionStmt;
    break;
  
  case Stmt::CXXCatchStmtClass:
    K = CXCursor_CXXCatchStmt;
    break;
  
  case Stmt::CXXTryStmtClass:
    K = CXCursor_CXXTryStmt;
    break;
  
  case Stmt::CXXForRangeStmtClass:
    K = CXCursor_CXXForRangeStmt;
    break;
  
  case Stmt::SEHTryStmtClass:
    K = CXCursor_SEHTryStmt;
    break;
  
  case Stmt::SEHExceptStmtClass:
    K = CXCursor_SEHExceptStmt;
    break;
  
  case Stmt::SEHFinallyStmtClass:
    K = CXCursor_SEHFinallyStmt;
    break;
  
  case Stmt::ArrayTypeTraitExprClass:
  case Stmt::AsTypeExprClass:
  case Stmt::AtomicExprClass:
  case Stmt::BinaryConditionalOperatorClass:
  case Stmt::BinaryTypeTraitExprClass:
  case Stmt::TypeTraitExprClass:
  case Stmt::CXXBindTemporaryExprClass:
  case Stmt::CXXDefaultArgExprClass:
  case Stmt::CXXScalarValueInitExprClass:
  case Stmt::CXXUuidofExprClass:
  case Stmt::ChooseExprClass:
  case Stmt::DesignatedInitExprClass:
  case Stmt::ExprWithCleanupsClass:
  case Stmt::ExpressionTraitExprClass:
  case Stmt::ExtVectorElementExprClass:
  case Stmt::ImplicitCastExprClass:
  case Stmt::ImplicitValueInitExprClass:
  case Stmt::MaterializeTemporaryExprClass:
  case Stmt::ObjCIndirectCopyRestoreExprClass:
  case Stmt::OffsetOfExprClass:
  case Stmt::ParenListExprClass:
  case Stmt::PredefinedExprClass:
  case Stmt::ShuffleVectorExprClass:
  case Stmt::UnaryExprOrTypeTraitExprClass:
  case Stmt::UnaryTypeTraitExprClass:
  case Stmt::VAArgExprClass:
  case Stmt::ObjCArrayLiteralClass:
  case Stmt::ObjCDictionaryLiteralClass:
  case Stmt::ObjCBoxedExprClass:
  case Stmt::ObjCSubscriptRefExprClass:
    K = CXCursor_UnexposedExpr;
    break;

  case Stmt::OpaqueValueExprClass:
    if (Expr *Src = cast<OpaqueValueExpr>(S)->getSourceExpr())
      return MakeCXCursor(Src, Parent, TU, RegionOfInterest);
    K = CXCursor_UnexposedExpr;
    break;

  case Stmt::PseudoObjectExprClass:
    return MakeCXCursor(cast<PseudoObjectExpr>(S)->getSyntacticForm(),
                        Parent, TU, RegionOfInterest);

  case Stmt::CompoundStmtClass:
    K = CXCursor_CompoundStmt;
    break;

  case Stmt::NullStmtClass:
    K = CXCursor_NullStmt;
    break;

  case Stmt::LabelStmtClass:
    K = CXCursor_LabelStmt;
    break;

  case Stmt::AttributedStmtClass:
    K = CXCursor_UnexposedStmt;
    break;

  case Stmt::DeclStmtClass:
    K = CXCursor_DeclStmt;
    break;

  case Stmt::IntegerLiteralClass:
    K = CXCursor_IntegerLiteral;
    break;

  case Stmt::FloatingLiteralClass:
    K = CXCursor_FloatingLiteral;
    break;

  case Stmt::ImaginaryLiteralClass:
    K = CXCursor_ImaginaryLiteral;
    break;

  case Stmt::StringLiteralClass:
    K = CXCursor_StringLiteral;
    break;

  case Stmt::CharacterLiteralClass:
    K = CXCursor_CharacterLiteral;
    break;

  case Stmt::ParenExprClass:
    K = CXCursor_ParenExpr;
    break;

  case Stmt::UnaryOperatorClass:
    K = CXCursor_UnaryOperator;
    break;

  case Stmt::CXXNoexceptExprClass:
    K = CXCursor_UnaryExpr;
    break;

  case Stmt::ArraySubscriptExprClass:
    K = CXCursor_ArraySubscriptExpr;
    break;

  case Stmt::BinaryOperatorClass:
    K = CXCursor_BinaryOperator;
    break;

  case Stmt::CompoundAssignOperatorClass:
    K = CXCursor_CompoundAssignOperator;
    break;

  case Stmt::ConditionalOperatorClass:
    K = CXCursor_ConditionalOperator;
    break;

  case Stmt::CStyleCastExprClass:
    K = CXCursor_CStyleCastExpr;
    break;

  case Stmt::CompoundLiteralExprClass:
    K = CXCursor_CompoundLiteralExpr;
    break;

  case Stmt::InitListExprClass:
    K = CXCursor_InitListExpr;
    break;

  case Stmt::AddrLabelExprClass:
    K = CXCursor_AddrLabelExpr;
    break;

  case Stmt::StmtExprClass:
    K = CXCursor_StmtExpr;
    break;

  case Stmt::GenericSelectionExprClass:
    K = CXCursor_GenericSelectionExpr;
    break;

  case Stmt::GNUNullExprClass:
    K = CXCursor_GNUNullExpr;
    break;

  case Stmt::CXXStaticCastExprClass:
    K = CXCursor_CXXStaticCastExpr;
    break;

  case Stmt::CXXDynamicCastExprClass:
    K = CXCursor_CXXDynamicCastExpr;
    break;

  case Stmt::CXXReinterpretCastExprClass:
    K = CXCursor_CXXReinterpretCastExpr;
    break;

  case Stmt::CXXConstCastExprClass:
    K = CXCursor_CXXConstCastExpr;
    break;

  case Stmt::CXXFunctionalCastExprClass:
    K = CXCursor_CXXFunctionalCastExpr;
    break;

  case Stmt::CXXTypeidExprClass:
    K = CXCursor_CXXTypeidExpr;
    break;

  case Stmt::CXXBoolLiteralExprClass:
    K = CXCursor_CXXBoolLiteralExpr;
    break;

  case Stmt::CXXNullPtrLiteralExprClass:
    K = CXCursor_CXXNullPtrLiteralExpr;
    break;

  case Stmt::CXXThisExprClass:
    K = CXCursor_CXXThisExpr;
    break;

  case Stmt::CXXThrowExprClass:
    K = CXCursor_CXXThrowExpr;
    break;

  case Stmt::CXXNewExprClass:
    K = CXCursor_CXXNewExpr;
    break;

  case Stmt::CXXDeleteExprClass:
    K = CXCursor_CXXDeleteExpr;
    break;

  case Stmt::ObjCStringLiteralClass:
    K = CXCursor_ObjCStringLiteral;
    break;

  case Stmt::ObjCEncodeExprClass:
    K = CXCursor_ObjCEncodeExpr;
    break;

  case Stmt::ObjCSelectorExprClass:
    K = CXCursor_ObjCSelectorExpr;
    break;

  case Stmt::ObjCProtocolExprClass:
    K = CXCursor_ObjCProtocolExpr;
    break;
      
  case Stmt::ObjCBoolLiteralExprClass:
    K = CXCursor_ObjCBoolLiteralExpr;
    break;
      
  case Stmt::ObjCBridgedCastExprClass:
    K = CXCursor_ObjCBridgedCastExpr;
    break;

  case Stmt::BlockExprClass:
    K = CXCursor_BlockExpr;
    break;

  case Stmt::PackExpansionExprClass:
    K = CXCursor_PackExpansionExpr;
    break;

  case Stmt::SizeOfPackExprClass:
    K = CXCursor_SizeOfPackExpr;
    break;

  case Stmt::DeclRefExprClass:           
  case Stmt::DependentScopeDeclRefExprClass:
  case Stmt::SubstNonTypeTemplateParmExprClass:
  case Stmt::SubstNonTypeTemplateParmPackExprClass:
  case Stmt::FunctionParmPackExprClass:
  case Stmt::UnresolvedLookupExprClass:
    K = CXCursor_DeclRefExpr;
    break;
      
  case Stmt::CXXDependentScopeMemberExprClass:
  case Stmt::CXXPseudoDestructorExprClass:
  case Stmt::MemberExprClass:            
  case Stmt::ObjCIsaExprClass:
  case Stmt::ObjCIvarRefExprClass:    
  case Stmt::ObjCPropertyRefExprClass: 
  case Stmt::UnresolvedMemberExprClass:
    K = CXCursor_MemberRefExpr;
    break;
      
  case Stmt::CallExprClass:              
  case Stmt::CXXOperatorCallExprClass:
  case Stmt::CXXMemberCallExprClass:
  case Stmt::CUDAKernelCallExprClass:
  case Stmt::CXXConstructExprClass:  
  case Stmt::CXXTemporaryObjectExprClass:
  case Stmt::CXXUnresolvedConstructExprClass:
  case Stmt::UserDefinedLiteralClass:
    K = CXCursor_CallExpr;
    break;
      
  case Stmt::LambdaExprClass:
    K = CXCursor_LambdaExpr;
    break;
      
  case Stmt::ObjCMessageExprClass: {
    K = CXCursor_ObjCMessageExpr;
    int SelectorIdIndex = -1;
    // Check if cursor points to a selector id.
    if (RegionOfInterest.isValid() &&
        RegionOfInterest.getBegin() == RegionOfInterest.getEnd()) {
      SmallVector<SourceLocation, 16> SelLocs;
      cast<ObjCMessageExpr>(S)->getSelectorLocs(SelLocs);
      SmallVector<SourceLocation, 16>::iterator
        I=std::find(SelLocs.begin(), SelLocs.end(),RegionOfInterest.getBegin());
      if (I != SelLocs.end())
        SelectorIdIndex = I - SelLocs.begin();
    }
    CXCursor C = { K, 0, { Parent, S, TU } };
    return getSelectorIdentifierCursor(SelectorIdIndex, C);
  }
      
  case Stmt::MSDependentExistsStmtClass:
    K = CXCursor_UnexposedStmt;
    break;
  }
  
  CXCursor C = { K, 0, { Parent, S, TU } };
  return C;
}

CXCursor cxcursor::MakeCursorObjCSuperClassRef(ObjCInterfaceDecl *Super, 
                                               SourceLocation Loc, 
                                               CXTranslationUnit TU) {
  assert(Super && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCSuperClassRef, 0, { Super, RawLoc, TU } };
  return C;    
}

std::pair<ObjCInterfaceDecl *, SourceLocation> 
cxcursor::getCursorObjCSuperClassRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCSuperClassRef);
  return std::make_pair(static_cast<ObjCInterfaceDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorObjCProtocolRef(const ObjCProtocolDecl *Proto, 
                                             SourceLocation Loc, 
                                             CXTranslationUnit TU) {
  assert(Proto && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCProtocolRef, 0, { (void*)Proto, RawLoc, TU } };
  return C;    
}

std::pair<ObjCProtocolDecl *, SourceLocation> 
cxcursor::getCursorObjCProtocolRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCProtocolRef);
  return std::make_pair(static_cast<ObjCProtocolDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorObjCClassRef(const ObjCInterfaceDecl *Class, 
                                          SourceLocation Loc, 
                                          CXTranslationUnit TU) {
  // 'Class' can be null for invalid code.
  if (!Class)
    return MakeCXCursorInvalid(CXCursor_InvalidCode);
  assert(TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCClassRef, 0, { (void*)Class, RawLoc, TU } };
  return C;    
}

std::pair<ObjCInterfaceDecl *, SourceLocation> 
cxcursor::getCursorObjCClassRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCClassRef);
  return std::make_pair(static_cast<ObjCInterfaceDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorTypeRef(const TypeDecl *Type, SourceLocation Loc, 
                                     CXTranslationUnit TU) {
  assert(Type && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_TypeRef, 0, { (void*)Type, RawLoc, TU } };
  return C;    
}

std::pair<TypeDecl *, SourceLocation> 
cxcursor::getCursorTypeRef(CXCursor C) {
  assert(C.kind == CXCursor_TypeRef);
  return std::make_pair(static_cast<TypeDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorTemplateRef(const TemplateDecl *Template, 
                                         SourceLocation Loc,
                                         CXTranslationUnit TU) {
  assert(Template && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_TemplateRef, 0, { (void*)Template, RawLoc, TU } };
  return C;    
}

std::pair<TemplateDecl *, SourceLocation> 
cxcursor::getCursorTemplateRef(CXCursor C) {
  assert(C.kind == CXCursor_TemplateRef);
  return std::make_pair(static_cast<TemplateDecl *>(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                                       reinterpret_cast<uintptr_t>(C.data[1])));  
}

CXCursor cxcursor::MakeCursorNamespaceRef(const NamedDecl *NS,
                                          SourceLocation Loc, 
                                          CXTranslationUnit TU) {
  
  assert(NS && (isa<NamespaceDecl>(NS) || isa<NamespaceAliasDecl>(NS)) && TU &&
         "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_NamespaceRef, 0, { (void*)NS, RawLoc, TU } };
  return C;    
}

std::pair<NamedDecl *, SourceLocation> 
cxcursor::getCursorNamespaceRef(CXCursor C) {
  assert(C.kind == CXCursor_NamespaceRef);
  return std::make_pair(static_cast<NamedDecl *>(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                                       reinterpret_cast<uintptr_t>(C.data[1])));  
}

CXCursor cxcursor::MakeCursorVariableRef(const VarDecl *Var, SourceLocation Loc, 
                                         CXTranslationUnit TU) {
  
  assert(Var && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_VariableRef, 0, { (void*)Var, RawLoc, TU } };
  return C;
}

std::pair<VarDecl *, SourceLocation> 
cxcursor::getCursorVariableRef(CXCursor C) {
  assert(C.kind == CXCursor_VariableRef);
  return std::make_pair(static_cast<VarDecl *>(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                          reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorMemberRef(const FieldDecl *Field, SourceLocation Loc, 
                                       CXTranslationUnit TU) {
  
  assert(Field && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_MemberRef, 0, { (void*)Field, RawLoc, TU } };
  return C;    
}

std::pair<FieldDecl *, SourceLocation> 
cxcursor::getCursorMemberRef(CXCursor C) {
  assert(C.kind == CXCursor_MemberRef);
  return std::make_pair(static_cast<FieldDecl *>(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                                       reinterpret_cast<uintptr_t>(C.data[1])));  
}

CXCursor cxcursor::MakeCursorCXXBaseSpecifier(const CXXBaseSpecifier *B,
                                              CXTranslationUnit TU){
  CXCursor C = { CXCursor_CXXBaseSpecifier, 0, { (void*)B, 0, TU } };
  return C;  
}

CXXBaseSpecifier *cxcursor::getCursorCXXBaseSpecifier(CXCursor C) {
  assert(C.kind == CXCursor_CXXBaseSpecifier);
  return static_cast<CXXBaseSpecifier*>(C.data[0]);
}

CXCursor cxcursor::MakePreprocessingDirectiveCursor(SourceRange Range, 
                                                    CXTranslationUnit TU) {
  CXCursor C = { CXCursor_PreprocessingDirective, 0,
                 { reinterpret_cast<void *>(Range.getBegin().getRawEncoding()),
                   reinterpret_cast<void *>(Range.getEnd().getRawEncoding()),
                   TU }
               };
  return C;
}

SourceRange cxcursor::getCursorPreprocessingDirective(CXCursor C) {
  assert(C.kind == CXCursor_PreprocessingDirective);
  SourceRange Range = SourceRange(SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t> (C.data[0])),
                     SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t> (C.data[1])));
  ASTUnit *TU = getCursorASTUnit(C);
  return TU->mapRangeFromPreamble(Range);
}

CXCursor cxcursor::MakeMacroDefinitionCursor(MacroDefinition *MI,
                                             CXTranslationUnit TU) {
  CXCursor C = { CXCursor_MacroDefinition, 0, { MI, 0, TU } };
  return C;
}

MacroDefinition *cxcursor::getCursorMacroDefinition(CXCursor C) {
  assert(C.kind == CXCursor_MacroDefinition);
  return static_cast<MacroDefinition *>(C.data[0]);
}

CXCursor cxcursor::MakeMacroExpansionCursor(MacroExpansion *MI, 
                                            CXTranslationUnit TU) {
  CXCursor C = { CXCursor_MacroExpansion, 0, { MI, 0, TU } };
  return C;
}

MacroExpansion *cxcursor::getCursorMacroExpansion(CXCursor C) {
  assert(C.kind == CXCursor_MacroExpansion);
  return static_cast<MacroExpansion *>(C.data[0]);
}

CXCursor cxcursor::MakeInclusionDirectiveCursor(InclusionDirective *ID, 
                                                CXTranslationUnit TU) {
  CXCursor C = { CXCursor_InclusionDirective, 0, { ID, 0, TU } };
  return C;
}

InclusionDirective *cxcursor::getCursorInclusionDirective(CXCursor C) {
  assert(C.kind == CXCursor_InclusionDirective);
  return static_cast<InclusionDirective *>(C.data[0]);  
}

CXCursor cxcursor::MakeCursorLabelRef(LabelStmt *Label, SourceLocation Loc, 
                                      CXTranslationUnit TU) {
  
  assert(Label && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_LabelRef, 0, { Label, RawLoc, TU } };
  return C;    
}

std::pair<LabelStmt*, SourceLocation> 
cxcursor::getCursorLabelRef(CXCursor C) {
  assert(C.kind == CXCursor_LabelRef);
  return std::make_pair(static_cast<LabelStmt *>(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                                       reinterpret_cast<uintptr_t>(C.data[1])));  
}

CXCursor cxcursor::MakeCursorOverloadedDeclRef(OverloadExpr *E, 
                                               CXTranslationUnit TU) {
  assert(E && TU && "Invalid arguments!");
  OverloadedDeclRefStorage Storage(E);
  void *RawLoc = reinterpret_cast<void *>(E->getNameLoc().getRawEncoding());
  CXCursor C = { 
                 CXCursor_OverloadedDeclRef, 0,
                 { Storage.getOpaqueValue(), RawLoc, TU } 
               };
  return C;    
}

CXCursor cxcursor::MakeCursorOverloadedDeclRef(Decl *D, 
                                               SourceLocation Loc,
                                               CXTranslationUnit TU) {
  assert(D && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  OverloadedDeclRefStorage Storage(D);
  CXCursor C = { 
    CXCursor_OverloadedDeclRef, 0,
    { Storage.getOpaqueValue(), RawLoc, TU }
  };
  return C;    
}

CXCursor cxcursor::MakeCursorOverloadedDeclRef(TemplateName Name, 
                                               SourceLocation Loc,
                                               CXTranslationUnit TU) {
  assert(Name.getAsOverloadedTemplate() && TU && "Invalid arguments!");
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  OverloadedDeclRefStorage Storage(Name.getAsOverloadedTemplate());
  CXCursor C = { 
    CXCursor_OverloadedDeclRef, 0,
    { Storage.getOpaqueValue(), RawLoc, TU } 
  };
  return C;    
}

std::pair<cxcursor::OverloadedDeclRefStorage, SourceLocation>
cxcursor::getCursorOverloadedDeclRef(CXCursor C) {
  assert(C.kind == CXCursor_OverloadedDeclRef);
  return std::make_pair(OverloadedDeclRefStorage::getFromOpaqueValue(C.data[0]),
                        SourceLocation::getFromRawEncoding(
                                       reinterpret_cast<uintptr_t>(C.data[1])));
}

Decl *cxcursor::getCursorDecl(CXCursor Cursor) {
  return (Decl *)Cursor.data[0];
}

Expr *cxcursor::getCursorExpr(CXCursor Cursor) {
  return dyn_cast_or_null<Expr>(getCursorStmt(Cursor));
}

Stmt *cxcursor::getCursorStmt(CXCursor Cursor) {
  if (Cursor.kind == CXCursor_ObjCSuperClassRef ||
      Cursor.kind == CXCursor_ObjCProtocolRef ||
      Cursor.kind == CXCursor_ObjCClassRef)
    return 0;

  return (Stmt *)Cursor.data[1];
}

Attr *cxcursor::getCursorAttr(CXCursor Cursor) {
  return (Attr *)Cursor.data[1];
}

Decl *cxcursor::getCursorParentDecl(CXCursor Cursor) {
  return (Decl *)Cursor.data[0];
}

ASTContext &cxcursor::getCursorContext(CXCursor Cursor) {
  return getCursorASTUnit(Cursor)->getASTContext();
}

ASTUnit *cxcursor::getCursorASTUnit(CXCursor Cursor) {
  CXTranslationUnit TU = static_cast<CXTranslationUnit>(Cursor.data[2]);
  if (!TU)
    return 0;
  return static_cast<ASTUnit *>(TU->TUData);
}

CXTranslationUnit cxcursor::getCursorTU(CXCursor Cursor) {
  return static_cast<CXTranslationUnit>(Cursor.data[2]);
}

void cxcursor::getOverriddenCursors(CXCursor cursor,
                                    SmallVectorImpl<CXCursor> &overridden) { 
  assert(clang_isDeclaration(cursor.kind));
  const NamedDecl *D = dyn_cast_or_null<NamedDecl>(getCursorDecl(cursor));
  if (!D)
    return;

  CXTranslationUnit TU = getCursorTU(cursor);
  SmallVector<const NamedDecl *, 8> OverDecls;
  D->getASTContext().getOverriddenMethods(D, OverDecls);

  for (SmallVector<const NamedDecl *, 8>::iterator
         I = OverDecls.begin(), E = OverDecls.end(); I != E; ++I) {
    overridden.push_back(MakeCXCursor(const_cast<NamedDecl*>(*I), TU));
  }
}

std::pair<int, SourceLocation>
cxcursor::getSelectorIdentifierIndexAndLoc(CXCursor cursor) {
  if (cursor.kind == CXCursor_ObjCMessageExpr) {
    if (cursor.xdata != -1)
      return std::make_pair(cursor.xdata,
                            cast<ObjCMessageExpr>(getCursorExpr(cursor))
                                                ->getSelectorLoc(cursor.xdata));
  } else if (cursor.kind == CXCursor_ObjCClassMethodDecl ||
             cursor.kind == CXCursor_ObjCInstanceMethodDecl) {
    if (cursor.xdata != -1)
      return std::make_pair(cursor.xdata,
                            cast<ObjCMethodDecl>(getCursorDecl(cursor))
                                                ->getSelectorLoc(cursor.xdata));
  }

  return std::make_pair(-1, SourceLocation());
}

CXCursor cxcursor::getSelectorIdentifierCursor(int SelIdx, CXCursor cursor) {
  CXCursor newCursor = cursor;

  if (cursor.kind == CXCursor_ObjCMessageExpr) {
    if (SelIdx == -1 ||
        unsigned(SelIdx) >= cast<ObjCMessageExpr>(getCursorExpr(cursor))
                                                         ->getNumSelectorLocs())
      newCursor.xdata = -1;
    else
      newCursor.xdata = SelIdx;
  } else if (cursor.kind == CXCursor_ObjCClassMethodDecl ||
             cursor.kind == CXCursor_ObjCInstanceMethodDecl) {
    if (SelIdx == -1 ||
        unsigned(SelIdx) >= cast<ObjCMethodDecl>(getCursorDecl(cursor))
                                                         ->getNumSelectorLocs())
      newCursor.xdata = -1;
    else
      newCursor.xdata = SelIdx;
  }

  return newCursor;
}

CXCursor cxcursor::getTypeRefCursor(CXCursor cursor) {
  if (cursor.kind != CXCursor_CallExpr)
    return cursor;

  if (cursor.xdata == 0)
    return cursor;

  Expr *E = getCursorExpr(cursor);
  TypeSourceInfo *Type = 0;
  if (CXXUnresolvedConstructExpr *
        UnCtor = dyn_cast<CXXUnresolvedConstructExpr>(E)) {
    Type = UnCtor->getTypeSourceInfo();
  } else if (CXXTemporaryObjectExpr *Tmp = dyn_cast<CXXTemporaryObjectExpr>(E)){
    Type = Tmp->getTypeSourceInfo();
  }

  if (!Type)
    return cursor;

  CXTranslationUnit TU = getCursorTU(cursor);
  QualType Ty = Type->getType();
  TypeLoc TL = Type->getTypeLoc();
  SourceLocation Loc = TL.getBeginLoc();

  if (const ElaboratedType *ElabT = Ty->getAs<ElaboratedType>()) {
    Ty = ElabT->getNamedType();
    ElaboratedTypeLoc ElabTL = cast<ElaboratedTypeLoc>(TL);
    Loc = ElabTL.getNamedTypeLoc().getBeginLoc();
  }

  if (const TypedefType *Typedef = Ty->getAs<TypedefType>())
    return MakeCursorTypeRef(Typedef->getDecl(), Loc, TU);
  if (const TagType *Tag = Ty->getAs<TagType>())
    return MakeCursorTypeRef(Tag->getDecl(), Loc, TU);
  if (const TemplateTypeParmType *TemplP = Ty->getAs<TemplateTypeParmType>())
    return MakeCursorTypeRef(TemplP->getDecl(), Loc, TU);

  return cursor;
}

bool cxcursor::operator==(CXCursor X, CXCursor Y) {
  return X.kind == Y.kind && X.data[0] == Y.data[0] && X.data[1] == Y.data[1] &&
         X.data[2] == Y.data[2];
}

// FIXME: Remove once we can model DeclGroups and their appropriate ranges
// properly in the ASTs.
bool cxcursor::isFirstInDeclGroup(CXCursor C) {
  assert(clang_isDeclaration(C.kind));
  return ((uintptr_t) (C.data[1])) != 0;
}

//===----------------------------------------------------------------------===//
// libclang CXCursor APIs
//===----------------------------------------------------------------------===//

extern "C" {

int clang_Cursor_isNull(CXCursor cursor) {
  return clang_equalCursors(cursor, clang_getNullCursor());
}

CXTranslationUnit clang_Cursor_getTranslationUnit(CXCursor cursor) {
  return getCursorTU(cursor);
}

int clang_Cursor_getNumArguments(CXCursor C) {
  if (clang_isDeclaration(C.kind)) {
    Decl *D = cxcursor::getCursorDecl(C);
    if (const ObjCMethodDecl *MD = dyn_cast_or_null<ObjCMethodDecl>(D))
      return MD->param_size();
    if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D))
      return FD->param_size();
  }

  return -1;
}

CXCursor clang_Cursor_getArgument(CXCursor C, unsigned i) {
  if (clang_isDeclaration(C.kind)) {
    Decl *D = cxcursor::getCursorDecl(C);
    if (ObjCMethodDecl *MD = dyn_cast_or_null<ObjCMethodDecl>(D)) {
      if (i < MD->param_size())
        return cxcursor::MakeCXCursor(MD->param_begin()[i],
                                      cxcursor::getCursorTU(C));
    } else if (FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D)) {
      if (i < FD->param_size())
        return cxcursor::MakeCXCursor(FD->param_begin()[i],
                                      cxcursor::getCursorTU(C));
    }
  }

  return clang_getNullCursor();
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXCursorSet.
//===----------------------------------------------------------------------===//

typedef llvm::DenseMap<CXCursor, unsigned> CXCursorSet_Impl;

static inline CXCursorSet packCXCursorSet(CXCursorSet_Impl *setImpl) {
  return (CXCursorSet) setImpl;
}
static inline CXCursorSet_Impl *unpackCXCursorSet(CXCursorSet set) {
  return (CXCursorSet_Impl*) set;
}
namespace llvm {
template<> struct DenseMapInfo<CXCursor> {
public:
  static inline CXCursor getEmptyKey() {
    return MakeCXCursorInvalid(CXCursor_InvalidFile);
  }
  static inline CXCursor getTombstoneKey() {
    return MakeCXCursorInvalid(CXCursor_NoDeclFound);
  }
  static inline unsigned getHashValue(const CXCursor &cursor) {
    return llvm::DenseMapInfo<std::pair<void*,void*> >
      ::getHashValue(std::make_pair(cursor.data[0], cursor.data[1]));
  }
  static inline bool isEqual(const CXCursor &x, const CXCursor &y) {
    return x.kind == y.kind &&
           x.data[0] == y.data[0] &&
           x.data[1] == y.data[1];
  }
};
}

extern "C" {
CXCursorSet clang_createCXCursorSet() {
  return packCXCursorSet(new CXCursorSet_Impl());
}

void clang_disposeCXCursorSet(CXCursorSet set) {
  delete unpackCXCursorSet(set);
}

unsigned clang_CXCursorSet_contains(CXCursorSet set, CXCursor cursor) {
  CXCursorSet_Impl *setImpl = unpackCXCursorSet(set);
  if (!setImpl)
    return 0;
  return setImpl->find(cursor) == setImpl->end();
}

unsigned clang_CXCursorSet_insert(CXCursorSet set, CXCursor cursor) {
  // Do not insert invalid cursors into the set.
  if (cursor.kind >= CXCursor_FirstInvalid &&
      cursor.kind <= CXCursor_LastInvalid)
    return 1;

  CXCursorSet_Impl *setImpl = unpackCXCursorSet(set);
  if (!setImpl)
    return 1;
  unsigned &entry = (*setImpl)[cursor];
  unsigned flag = entry == 0 ? 1 : 0;
  entry = 1;
  return flag;
}
  
CXCompletionString clang_getCursorCompletionString(CXCursor cursor) {
  enum CXCursorKind kind = clang_getCursorKind(cursor);
  if (clang_isDeclaration(kind)) {
    Decl *decl = getCursorDecl(cursor);
    if (NamedDecl *namedDecl = dyn_cast_or_null<NamedDecl>(decl)) {
      ASTUnit *unit = getCursorASTUnit(cursor);
      CodeCompletionResult Result(namedDecl);
      CodeCompletionString *String
        = Result.CreateCodeCompletionString(unit->getASTContext(),
                                            unit->getPreprocessor(),
                                 unit->getCodeCompletionTUInfo().getAllocator(),
                                 unit->getCodeCompletionTUInfo(),
                                 true);
      return String;
    }
  }
  else if (kind == CXCursor_MacroDefinition) {
    MacroDefinition *definition = getCursorMacroDefinition(cursor);
    const IdentifierInfo *MacroInfo = definition->getName();
    ASTUnit *unit = getCursorASTUnit(cursor);
    CodeCompletionResult Result(const_cast<IdentifierInfo *>(MacroInfo));
    CodeCompletionString *String
      = Result.CreateCodeCompletionString(unit->getASTContext(),
                                          unit->getPreprocessor(),
                                 unit->getCodeCompletionTUInfo().getAllocator(),
                                 unit->getCodeCompletionTUInfo(),
                                 false);
    return String;
  }
  return NULL;
}
} // end: extern C.

namespace {
  struct OverridenCursorsPool {
    typedef llvm::SmallVector<CXCursor, 2> CursorVec;
    std::vector<CursorVec*> AllCursors;
    std::vector<CursorVec*> AvailableCursors;
    
    ~OverridenCursorsPool() {
      for (std::vector<CursorVec*>::iterator I = AllCursors.begin(),
           E = AllCursors.end(); I != E; ++I) {
        delete *I;
      }
    }
  };
}

void *cxcursor::createOverridenCXCursorsPool() {
  return new OverridenCursorsPool();
}
  
void cxcursor::disposeOverridenCXCursorsPool(void *pool) {
  delete static_cast<OverridenCursorsPool*>(pool);
}
 
extern "C" {
void clang_getOverriddenCursors(CXCursor cursor,
                                CXCursor **overridden,
                                unsigned *num_overridden) {
  if (overridden)
    *overridden = 0;
  if (num_overridden)
    *num_overridden = 0;
  
  CXTranslationUnit TU = cxcursor::getCursorTU(cursor);
  
  if (!overridden || !num_overridden || !TU)
    return;

  if (!clang_isDeclaration(cursor.kind))
    return;
    
  OverridenCursorsPool &pool =
    *static_cast<OverridenCursorsPool*>(TU->OverridenCursorsPool);
  
  OverridenCursorsPool::CursorVec *Vec = 0;
  
  if (!pool.AvailableCursors.empty()) {
    Vec = pool.AvailableCursors.back();
    pool.AvailableCursors.pop_back();
  }
  else {
    Vec = new OverridenCursorsPool::CursorVec();
    pool.AllCursors.push_back(Vec);
  }
  
  // Clear out the vector, but don't free the memory contents.  This
  // reduces malloc() traffic.
  Vec->clear();

  // Use the first entry to contain a back reference to the vector.
  // This is a complete hack.
  CXCursor backRefCursor = MakeCXCursorInvalid(CXCursor_InvalidFile, TU);
  backRefCursor.data[0] = Vec;
  assert(cxcursor::getCursorTU(backRefCursor) == TU);
  Vec->push_back(backRefCursor);

  // Get the overriden cursors.
  cxcursor::getOverriddenCursors(cursor, *Vec);
  
  // Did we get any overriden cursors?  If not, return Vec to the pool
  // of available cursor vectors.
  if (Vec->size() == 1) {
    pool.AvailableCursors.push_back(Vec);
    return;
  }

  // Now tell the caller about the overriden cursors.
  assert(Vec->size() > 1);
  *overridden = &((*Vec)[1]);
  *num_overridden = Vec->size() - 1;
}

void clang_disposeOverriddenCursors(CXCursor *overridden) {
  if (!overridden)
    return;
  
  // Use pointer arithmetic to get back the first faux entry
  // which has a back-reference to the TU and the vector.
  --overridden;
  OverridenCursorsPool::CursorVec *Vec =
    static_cast<OverridenCursorsPool::CursorVec*>(overridden->data[0]);
  CXTranslationUnit TU = getCursorTU(*overridden);
  
  assert(Vec && TU);

  OverridenCursorsPool &pool =
    *static_cast<OverridenCursorsPool*>(TU->OverridenCursorsPool);
  
  pool.AvailableCursors.push_back(Vec);
}

int clang_Cursor_isDynamicCall(CXCursor C) {
  const Expr *E = 0;
  if (clang_isExpression(C.kind))
    E = getCursorExpr(C);
  if (!E)
    return 0;

  if (const ObjCMessageExpr *MsgE = dyn_cast<ObjCMessageExpr>(E))
    return MsgE->getReceiverKind() == ObjCMessageExpr::Instance;

  const MemberExpr *ME = 0;
  if (isa<MemberExpr>(E))
    ME = cast<MemberExpr>(E);
  else if (const CallExpr *CE = dyn_cast<CallExpr>(E))
    ME = dyn_cast_or_null<MemberExpr>(CE->getCallee());

  if (ME) {
    if (const CXXMethodDecl *
          MD = dyn_cast_or_null<CXXMethodDecl>(ME->getMemberDecl()))
      return MD->isVirtual() && !ME->hasQualifier();
  }

  return 0;
}

CXType clang_Cursor_getReceiverType(CXCursor C) {
  CXTranslationUnit TU = cxcursor::getCursorTU(C);
  const Expr *E = 0;
  if (clang_isExpression(C.kind))
    E = getCursorExpr(C);

  if (const ObjCMessageExpr *MsgE = dyn_cast_or_null<ObjCMessageExpr>(E))
    return cxtype::MakeCXType(MsgE->getReceiverType(), TU);

  return cxtype::MakeCXType(QualType(), TU);
}

} // end: extern "C"
