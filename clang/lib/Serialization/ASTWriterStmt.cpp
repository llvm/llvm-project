//===--- ASTWriterStmt.cpp - Statement and Expression Serialization -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements serialization for Statements and Expressions.
///
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTWriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Token.h"
#include "llvm/Bitcode/BitstreamWriter.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Statement/expression serialization
//===----------------------------------------------------------------------===//

namespace clang {

  class ASTStmtWriter : public StmtVisitor<ASTStmtWriter, void> {
    friend class OMPClauseWriter;
    ASTWriter &Writer;
    ASTWriter::RecordData &Record;

  public:
    serialization::StmtCode Code;
    unsigned AbbrevToUse;

    ASTStmtWriter(ASTWriter &Writer, ASTWriter::RecordData &Record)
      : Writer(Writer), Record(Record) { }

    void AddTemplateKWAndArgsInfo(const ASTTemplateKWAndArgsInfo &ArgInfo,
                                  const TemplateArgumentLoc *Args);

    void VisitStmt(Stmt *S);
#define STMT(Type, Base) \
    void Visit##Type(Type *);
#include "clang/AST/StmtNodes.inc"
  };
}

void ASTStmtWriter::AddTemplateKWAndArgsInfo(
    const ASTTemplateKWAndArgsInfo &ArgInfo, const TemplateArgumentLoc *Args) {
  Writer.AddSourceLocation(ArgInfo.TemplateKWLoc, Record);
  Writer.AddSourceLocation(ArgInfo.LAngleLoc, Record);
  Writer.AddSourceLocation(ArgInfo.RAngleLoc, Record);
  for (unsigned i = 0; i != ArgInfo.NumTemplateArgs; ++i)
    Writer.AddTemplateArgumentLoc(Args[i], Record);
}

void ASTStmtWriter::VisitStmt(Stmt *S) {
}

void ASTStmtWriter::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getSemiLoc(), Record);
  Record.push_back(S->HasLeadingEmptyMacro);
  Code = serialization::STMT_NULL;
}

void ASTStmtWriter::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
  Record.push_back(S->size());
  for (auto *CS : S->body())
    Writer.AddStmt(CS);
  Writer.AddSourceLocation(S->getLBracLoc(), Record);
  Writer.AddSourceLocation(S->getRBracLoc(), Record);
  Code = serialization::STMT_COMPOUND;
}

void ASTStmtWriter::VisitSwitchCase(SwitchCase *S) {
  VisitStmt(S);
  Record.push_back(Writer.getSwitchCaseID(S));
  Writer.AddSourceLocation(S->getKeywordLoc(), Record);
  Writer.AddSourceLocation(S->getColonLoc(), Record);
}

void ASTStmtWriter::VisitCaseStmt(CaseStmt *S) {
  VisitSwitchCase(S);
  Writer.AddStmt(S->getLHS());
  Writer.AddStmt(S->getRHS());
  Writer.AddStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getEllipsisLoc(), Record);
  Code = serialization::STMT_CASE;
}

void ASTStmtWriter::VisitDefaultStmt(DefaultStmt *S) {
  VisitSwitchCase(S);
  Writer.AddStmt(S->getSubStmt());
  Code = serialization::STMT_DEFAULT;
}

void ASTStmtWriter::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getDecl(), Record);
  Writer.AddStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getIdentLoc(), Record);
  Code = serialization::STMT_LABEL;
}

void ASTStmtWriter::VisitAttributedStmt(AttributedStmt *S) {
  VisitStmt(S);
  Record.push_back(S->getAttrs().size());
  Writer.WriteAttributes(S->getAttrs(), Record);
  Writer.AddStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getAttrLoc(), Record);
  Code = serialization::STMT_ATTRIBUTED;
}

void ASTStmtWriter::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.AddStmt(S->getCond());
  Writer.AddStmt(S->getThen());
  Writer.AddStmt(S->getElse());
  Writer.AddSourceLocation(S->getIfLoc(), Record);
  Writer.AddSourceLocation(S->getElseLoc(), Record);
  Code = serialization::STMT_IF;
}

void ASTStmtWriter::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.AddStmt(S->getCond());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getSwitchLoc(), Record);
  Record.push_back(S->isAllEnumCasesCovered());
  for (SwitchCase *SC = S->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase())
    Record.push_back(Writer.RecordSwitchCaseID(SC));
  Code = serialization::STMT_SWITCH;
}

void ASTStmtWriter::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.AddStmt(S->getCond());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getWhileLoc(), Record);
  Code = serialization::STMT_WHILE;
}

void ASTStmtWriter::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
  Writer.AddStmt(S->getCond());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getDoLoc(), Record);
  Writer.AddSourceLocation(S->getWhileLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = serialization::STMT_DO;
}

void ASTStmtWriter::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
  Writer.AddStmt(S->getInit());
  Writer.AddStmt(S->getCond());
  Writer.AddDeclRef(S->getConditionVariable(), Record);
  Writer.AddStmt(S->getInc());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getForLoc(), Record);
  Writer.AddSourceLocation(S->getLParenLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = serialization::STMT_FOR;
}

void ASTStmtWriter::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  Writer.AddDeclRef(S->getLabel(), Record);
  Writer.AddSourceLocation(S->getGotoLoc(), Record);
  Writer.AddSourceLocation(S->getLabelLoc(), Record);
  Code = serialization::STMT_GOTO;
}

void ASTStmtWriter::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getGotoLoc(), Record);
  Writer.AddSourceLocation(S->getStarLoc(), Record);
  Writer.AddStmt(S->getTarget());
  Code = serialization::STMT_INDIRECT_GOTO;
}

void ASTStmtWriter::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getContinueLoc(), Record);
  Code = serialization::STMT_CONTINUE;
}

void ASTStmtWriter::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getBreakLoc(), Record);
  Code = serialization::STMT_BREAK;
}

void ASTStmtWriter::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
  Writer.AddStmt(S->getRetValue());
  Writer.AddSourceLocation(S->getReturnLoc(), Record);
  Writer.AddDeclRef(S->getNRVOCandidate(), Record);
  Code = serialization::STMT_RETURN;
}

void ASTStmtWriter::VisitDeclStmt(DeclStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getStartLoc(), Record);
  Writer.AddSourceLocation(S->getEndLoc(), Record);
  DeclGroupRef DG = S->getDeclGroup();
  for (DeclGroupRef::iterator D = DG.begin(), DEnd = DG.end(); D != DEnd; ++D)
    Writer.AddDeclRef(*D, Record);
  Code = serialization::STMT_DECL;
}

void ASTStmtWriter::VisitAsmStmt(AsmStmt *S) {
  VisitStmt(S);
  Record.push_back(S->getNumOutputs());
  Record.push_back(S->getNumInputs());
  Record.push_back(S->getNumClobbers());
  Writer.AddSourceLocation(S->getAsmLoc(), Record);
  Record.push_back(S->isVolatile());
  Record.push_back(S->isSimple());
}

void ASTStmtWriter::VisitGCCAsmStmt(GCCAsmStmt *S) {
  VisitAsmStmt(S);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Writer.AddStmt(S->getAsmString());

  // Outputs
  for (unsigned I = 0, N = S->getNumOutputs(); I != N; ++I) {      
    Writer.AddIdentifierRef(S->getOutputIdentifier(I), Record);
    Writer.AddStmt(S->getOutputConstraintLiteral(I));
    Writer.AddStmt(S->getOutputExpr(I));
  }

  // Inputs
  for (unsigned I = 0, N = S->getNumInputs(); I != N; ++I) {
    Writer.AddIdentifierRef(S->getInputIdentifier(I), Record);
    Writer.AddStmt(S->getInputConstraintLiteral(I));
    Writer.AddStmt(S->getInputExpr(I));
  }

  // Clobbers
  for (unsigned I = 0, N = S->getNumClobbers(); I != N; ++I)
    Writer.AddStmt(S->getClobberStringLiteral(I));

  Code = serialization::STMT_GCCASM;
}

void ASTStmtWriter::VisitMSAsmStmt(MSAsmStmt *S) {
  VisitAsmStmt(S);
  Writer.AddSourceLocation(S->getLBraceLoc(), Record);
  Writer.AddSourceLocation(S->getEndLoc(), Record);
  Record.push_back(S->getNumAsmToks());
  Writer.AddString(S->getAsmString(), Record);

  // Tokens
  for (unsigned I = 0, N = S->getNumAsmToks(); I != N; ++I) {
    Writer.AddToken(S->getAsmToks()[I], Record);
  }

  // Clobbers
  for (unsigned I = 0, N = S->getNumClobbers(); I != N; ++I) {
    Writer.AddString(S->getClobber(I), Record);
  }

  // Outputs
  for (unsigned I = 0, N = S->getNumOutputs(); I != N; ++I) {      
    Writer.AddStmt(S->getOutputExpr(I));
    Writer.AddString(S->getOutputConstraint(I), Record);
  }

  // Inputs
  for (unsigned I = 0, N = S->getNumInputs(); I != N; ++I) {
    Writer.AddStmt(S->getInputExpr(I));
    Writer.AddString(S->getInputConstraint(I), Record);
  }

  Code = serialization::STMT_MSASM;
}

void ASTStmtWriter::VisitCoroutineBodyStmt(CoroutineBodyStmt *S) {
  // FIXME: Implement coroutine serialization.
  llvm_unreachable("unimplemented");
}

void ASTStmtWriter::VisitCoreturnStmt(CoreturnStmt *S) {
  // FIXME: Implement coroutine serialization.
  llvm_unreachable("unimplemented");
}

void ASTStmtWriter::VisitCoawaitExpr(CoawaitExpr *S) {
  // FIXME: Implement coroutine serialization.
  llvm_unreachable("unimplemented");
}

void ASTStmtWriter::VisitCoyieldExpr(CoyieldExpr *S) {
  // FIXME: Implement coroutine serialization.
  llvm_unreachable("unimplemented");
}

void ASTStmtWriter::VisitCapturedStmt(CapturedStmt *S) {
  VisitStmt(S);
  // NumCaptures
  Record.push_back(std::distance(S->capture_begin(), S->capture_end()));

  // CapturedDecl and captured region kind
  Writer.AddDeclRef(S->getCapturedDecl(), Record);
  Record.push_back(S->getCapturedRegionKind());

  Writer.AddDeclRef(S->getCapturedRecordDecl(), Record);

  // Capture inits
  for (auto *I : S->capture_inits())
    Writer.AddStmt(I);

  // Body
  Writer.AddStmt(S->getCapturedStmt());

  // Captures
  for (const auto &I : S->captures()) {
    if (I.capturesThis() || I.capturesVariableArrayType())
      Writer.AddDeclRef(nullptr, Record);
    else
      Writer.AddDeclRef(I.getCapturedVar(), Record);
    Record.push_back(I.getCaptureKind());
    Writer.AddSourceLocation(I.getLocation(), Record);
  }

  Code = serialization::STMT_CAPTURED;
}

void ASTStmtWriter::VisitExpr(Expr *E) {
  VisitStmt(E);
  Writer.AddTypeRef(E->getType(), Record);
  Record.push_back(E->isTypeDependent());
  Record.push_back(E->isValueDependent());
  Record.push_back(E->isInstantiationDependent());
  Record.push_back(E->containsUnexpandedParameterPack());
  Record.push_back(E->getValueKind());
  Record.push_back(E->getObjectKind());
}

void ASTStmtWriter::VisitPredefinedExpr(PredefinedExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->getIdentType()); // FIXME: stable encoding
  Writer.AddStmt(E->getFunctionName());
  Code = serialization::EXPR_PREDEFINED;
}

void ASTStmtWriter::VisitDeclRefExpr(DeclRefExpr *E) {
  VisitExpr(E);

  Record.push_back(E->hasQualifier());
  Record.push_back(E->getDecl() != E->getFoundDecl());
  Record.push_back(E->hasTemplateKWAndArgsInfo());
  Record.push_back(E->hadMultipleCandidates());
  Record.push_back(E->refersToEnclosingVariableOrCapture());

  if (E->hasTemplateKWAndArgsInfo()) {
    unsigned NumTemplateArgs = E->getNumTemplateArgs();
    Record.push_back(NumTemplateArgs);
  }

  DeclarationName::NameKind nk = (E->getDecl()->getDeclName().getNameKind());

  if ((!E->hasTemplateKWAndArgsInfo()) && (!E->hasQualifier()) &&
      (E->getDecl() == E->getFoundDecl()) &&
      nk == DeclarationName::Identifier) {
    AbbrevToUse = Writer.getDeclRefExprAbbrev();
  }

  if (E->hasQualifier())
    Writer.AddNestedNameSpecifierLoc(E->getQualifierLoc(), Record);

  if (E->getDecl() != E->getFoundDecl())
    Writer.AddDeclRef(E->getFoundDecl(), Record);

  if (E->hasTemplateKWAndArgsInfo())
    AddTemplateKWAndArgsInfo(*E->getTrailingObjects<ASTTemplateKWAndArgsInfo>(),
                             E->getTrailingObjects<TemplateArgumentLoc>());

  Writer.AddDeclRef(E->getDecl(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddDeclarationNameLoc(E->DNLoc, E->getDecl()->getDeclName(), Record);
  Code = serialization::EXPR_DECL_REF;
}

void ASTStmtWriter::VisitIntegerLiteral(IntegerLiteral *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddAPInt(E->getValue(), Record);

  if (E->getValue().getBitWidth() == 32) {
    AbbrevToUse = Writer.getIntegerLiteralAbbrev();
  }

  Code = serialization::EXPR_INTEGER_LITERAL;
}

void ASTStmtWriter::VisitFloatingLiteral(FloatingLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getRawSemantics());
  Record.push_back(E->isExact());
  Writer.AddAPFloat(E->getValue(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = serialization::EXPR_FLOATING_LITERAL;
}

void ASTStmtWriter::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_IMAGINARY_LITERAL;
}

void ASTStmtWriter::VisitStringLiteral(StringLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getByteLength());
  Record.push_back(E->getNumConcatenated());
  Record.push_back(E->getKind());
  Record.push_back(E->isPascal());
  // FIXME: String data should be stored as a blob at the end of the
  // StringLiteral. However, we can't do so now because we have no
  // provision for coping with abbreviations when we're jumping around
  // the AST file during deserialization.
  Record.append(E->getBytes().begin(), E->getBytes().end());
  for (unsigned I = 0, N = E->getNumConcatenated(); I != N; ++I)
    Writer.AddSourceLocation(E->getStrTokenLoc(I), Record);
  Code = serialization::EXPR_STRING_LITERAL;
}

void ASTStmtWriter::VisitCharacterLiteral(CharacterLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getValue());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->getKind());

  AbbrevToUse = Writer.getCharacterLiteralAbbrev();

  Code = serialization::EXPR_CHARACTER_LITERAL;
}

void ASTStmtWriter::VisitParenExpr(ParenExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLParen(), Record);
  Writer.AddSourceLocation(E->getRParen(), Record);
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_PAREN;
}

void ASTStmtWriter::VisitParenListExpr(ParenListExpr *E) {
  VisitExpr(E);
  Record.push_back(E->NumExprs);
  for (unsigned i=0; i != E->NumExprs; ++i)
    Writer.AddStmt(E->Exprs[i]);
  Writer.AddSourceLocation(E->LParenLoc, Record);
  Writer.AddSourceLocation(E->RParenLoc, Record);
  Code = serialization::EXPR_PAREN_LIST;
}

void ASTStmtWriter::VisitUnaryOperator(UnaryOperator *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Record.push_back(E->getOpcode()); // FIXME: stable encoding
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Code = serialization::EXPR_UNARY_OPERATOR;
}

void ASTStmtWriter::VisitOffsetOfExpr(OffsetOfExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumComponents());
  Record.push_back(E->getNumExpressions());
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Writer.AddTypeSourceInfo(E->getTypeSourceInfo(), Record);
  for (unsigned I = 0, N = E->getNumComponents(); I != N; ++I) {
    const OffsetOfNode &ON = E->getComponent(I);
    Record.push_back(ON.getKind()); // FIXME: Stable encoding
    Writer.AddSourceLocation(ON.getSourceRange().getBegin(), Record);
    Writer.AddSourceLocation(ON.getSourceRange().getEnd(), Record);
    switch (ON.getKind()) {
    case OffsetOfNode::Array:
      Record.push_back(ON.getArrayExprIndex());
      break;

    case OffsetOfNode::Field:
      Writer.AddDeclRef(ON.getField(), Record);
      break;

    case OffsetOfNode::Identifier:
      Writer.AddIdentifierRef(ON.getFieldName(), Record);
      break;

    case OffsetOfNode::Base:
      Writer.AddCXXBaseSpecifier(*ON.getBase(), Record);
      break;
    }
  }
  for (unsigned I = 0, N = E->getNumExpressions(); I != N; ++I)
    Writer.AddStmt(E->getIndexExpr(I));
  Code = serialization::EXPR_OFFSETOF;
}

void ASTStmtWriter::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getKind());
  if (E->isArgumentType())
    Writer.AddTypeSourceInfo(E->getArgumentTypeInfo(), Record);
  else {
    Record.push_back(0);
    Writer.AddStmt(E->getArgumentExpr());
  }
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_SIZEOF_ALIGN_OF;
}

void ASTStmtWriter::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getLHS());
  Writer.AddStmt(E->getRHS());
  Writer.AddSourceLocation(E->getRBracketLoc(), Record);
  Code = serialization::EXPR_ARRAY_SUBSCRIPT;
}

void ASTStmtWriter::VisitOMPArraySectionExpr(OMPArraySectionExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getBase());
  Writer.AddStmt(E->getLowerBound());
  Writer.AddStmt(E->getLength());
  Writer.AddSourceLocation(E->getColonLoc(), Record);
  Writer.AddSourceLocation(E->getRBracketLoc(), Record);
  Code = serialization::EXPR_OMP_ARRAY_SECTION;
}

void ASTStmtWriter::VisitCallExpr(CallExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumArgs());
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Writer.AddStmt(E->getCallee());
  for (CallExpr::arg_iterator Arg = E->arg_begin(), ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg)
    Writer.AddStmt(*Arg);
  Code = serialization::EXPR_CALL;
}

void ASTStmtWriter::VisitMemberExpr(MemberExpr *E) {
  // Don't call VisitExpr, we'll write everything here.

  Record.push_back(E->hasQualifier());
  if (E->hasQualifier())
    Writer.AddNestedNameSpecifierLoc(E->getQualifierLoc(), Record);

  Record.push_back(E->HasTemplateKWAndArgsInfo);
  if (E->HasTemplateKWAndArgsInfo) {
    Writer.AddSourceLocation(E->getTemplateKeywordLoc(), Record);
    unsigned NumTemplateArgs = E->getNumTemplateArgs();
    Record.push_back(NumTemplateArgs);
    Writer.AddSourceLocation(E->getLAngleLoc(), Record);
    Writer.AddSourceLocation(E->getRAngleLoc(), Record);
    for (unsigned i=0; i != NumTemplateArgs; ++i)
      Writer.AddTemplateArgumentLoc(E->getTemplateArgs()[i], Record);
  }

  Record.push_back(E->hadMultipleCandidates());

  DeclAccessPair FoundDecl = E->getFoundDecl();
  Writer.AddDeclRef(FoundDecl.getDecl(), Record);
  Record.push_back(FoundDecl.getAccess());

  Writer.AddTypeRef(E->getType(), Record);
  Record.push_back(E->getValueKind());
  Record.push_back(E->getObjectKind());
  Writer.AddStmt(E->getBase());
  Writer.AddDeclRef(E->getMemberDecl(), Record);
  Writer.AddSourceLocation(E->getMemberLoc(), Record);
  Record.push_back(E->isArrow());
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddDeclarationNameLoc(E->MemberDNLoc,
                               E->getMemberDecl()->getDeclName(), Record);
  Code = serialization::EXPR_MEMBER;
}

void ASTStmtWriter::VisitObjCIsaExpr(ObjCIsaExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getBase());
  Writer.AddSourceLocation(E->getIsaMemberLoc(), Record);
  Writer.AddSourceLocation(E->getOpLoc(), Record);
  Record.push_back(E->isArrow());
  Code = serialization::EXPR_OBJC_ISA;
}

void ASTStmtWriter::
VisitObjCIndirectCopyRestoreExpr(ObjCIndirectCopyRestoreExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Record.push_back(E->shouldCopy());
  Code = serialization::EXPR_OBJC_INDIRECT_COPY_RESTORE;
}

void ASTStmtWriter::VisitObjCBridgedCastExpr(ObjCBridgedCastExpr *E) {
  VisitExplicitCastExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getBridgeKeywordLoc(), Record);
  Record.push_back(E->getBridgeKind()); // FIXME: Stable encoding
  Code = serialization::EXPR_OBJC_BRIDGED_CAST;
}

void ASTStmtWriter::VisitCastExpr(CastExpr *E) {
  VisitExpr(E);
  Record.push_back(E->path_size());
  Writer.AddStmt(E->getSubExpr());
  Record.push_back(E->getCastKind()); // FIXME: stable encoding

  for (CastExpr::path_iterator
         PI = E->path_begin(), PE = E->path_end(); PI != PE; ++PI)
    Writer.AddCXXBaseSpecifier(**PI, Record);
}

void ASTStmtWriter::VisitBinaryOperator(BinaryOperator *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getLHS());
  Writer.AddStmt(E->getRHS());
  Record.push_back(E->getOpcode()); // FIXME: stable encoding
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Record.push_back(E->isFPContractable());
  Code = serialization::EXPR_BINARY_OPERATOR;
}

void ASTStmtWriter::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  VisitBinaryOperator(E);
  Writer.AddTypeRef(E->getComputationLHSType(), Record);
  Writer.AddTypeRef(E->getComputationResultType(), Record);
  Code = serialization::EXPR_COMPOUND_ASSIGN_OPERATOR;
}

void ASTStmtWriter::VisitConditionalOperator(ConditionalOperator *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getCond());
  Writer.AddStmt(E->getLHS());
  Writer.AddStmt(E->getRHS());
  Writer.AddSourceLocation(E->getQuestionLoc(), Record);
  Writer.AddSourceLocation(E->getColonLoc(), Record);
  Code = serialization::EXPR_CONDITIONAL_OPERATOR;
}

void
ASTStmtWriter::VisitBinaryConditionalOperator(BinaryConditionalOperator *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getOpaqueValue());
  Writer.AddStmt(E->getCommon());
  Writer.AddStmt(E->getCond());
  Writer.AddStmt(E->getTrueExpr());
  Writer.AddStmt(E->getFalseExpr());
  Writer.AddSourceLocation(E->getQuestionLoc(), Record);
  Writer.AddSourceLocation(E->getColonLoc(), Record);
  Code = serialization::EXPR_BINARY_CONDITIONAL_OPERATOR;
}

void ASTStmtWriter::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  VisitCastExpr(E);

  if (E->path_size() == 0)
    AbbrevToUse = Writer.getExprImplicitCastAbbrev();

  Code = serialization::EXPR_IMPLICIT_CAST;
}

void ASTStmtWriter::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  VisitCastExpr(E);
  Writer.AddTypeSourceInfo(E->getTypeInfoAsWritten(), Record);
}

void ASTStmtWriter::VisitCStyleCastExpr(CStyleCastExpr *E) {
  VisitExplicitCastExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CSTYLE_CAST;
}

void ASTStmtWriter::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddTypeSourceInfo(E->getTypeSourceInfo(), Record);
  Writer.AddStmt(E->getInitializer());
  Record.push_back(E->isFileScope());
  Code = serialization::EXPR_COMPOUND_LITERAL;
}

void ASTStmtWriter::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getBase());
  Writer.AddIdentifierRef(&E->getAccessor(), Record);
  Writer.AddSourceLocation(E->getAccessorLoc(), Record);
  Code = serialization::EXPR_EXT_VECTOR_ELEMENT;
}

void ASTStmtWriter::VisitInitListExpr(InitListExpr *E) {
  VisitExpr(E);
  // NOTE: only add the (possibly null) syntactic form.
  // No need to serialize the isSemanticForm flag and the semantic form.
  Writer.AddStmt(E->getSyntacticForm());
  Writer.AddSourceLocation(E->getLBraceLoc(), Record);
  Writer.AddSourceLocation(E->getRBraceLoc(), Record);
  bool isArrayFiller = E->ArrayFillerOrUnionFieldInit.is<Expr*>();
  Record.push_back(isArrayFiller);
  if (isArrayFiller)
    Writer.AddStmt(E->getArrayFiller());
  else
    Writer.AddDeclRef(E->getInitializedFieldInUnion(), Record);
  Record.push_back(E->hadArrayRangeDesignator());
  Record.push_back(E->getNumInits());
  if (isArrayFiller) {
    // ArrayFiller may have filled "holes" due to designated initializer.
    // Replace them by 0 to indicate that the filler goes in that place.
    Expr *filler = E->getArrayFiller();
    for (unsigned I = 0, N = E->getNumInits(); I != N; ++I)
      Writer.AddStmt(E->getInit(I) != filler ? E->getInit(I) : nullptr);
  } else {
    for (unsigned I = 0, N = E->getNumInits(); I != N; ++I)
      Writer.AddStmt(E->getInit(I));
  }
  Code = serialization::EXPR_INIT_LIST;
}

void ASTStmtWriter::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumSubExprs());
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I)
    Writer.AddStmt(E->getSubExpr(I));
  Writer.AddSourceLocation(E->getEqualOrColonLoc(), Record);
  Record.push_back(E->usesGNUSyntax());
  for (DesignatedInitExpr::designators_iterator D = E->designators_begin(),
                                             DEnd = E->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      if (FieldDecl *Field = D->getField()) {
        Record.push_back(serialization::DESIG_FIELD_DECL);
        Writer.AddDeclRef(Field, Record);
      } else {
        Record.push_back(serialization::DESIG_FIELD_NAME);
        Writer.AddIdentifierRef(D->getFieldName(), Record);
      }
      Writer.AddSourceLocation(D->getDotLoc(), Record);
      Writer.AddSourceLocation(D->getFieldLoc(), Record);
    } else if (D->isArrayDesignator()) {
      Record.push_back(serialization::DESIG_ARRAY);
      Record.push_back(D->getFirstExprIndex());
      Writer.AddSourceLocation(D->getLBracketLoc(), Record);
      Writer.AddSourceLocation(D->getRBracketLoc(), Record);
    } else {
      assert(D->isArrayRangeDesignator() && "Unknown designator");
      Record.push_back(serialization::DESIG_ARRAY_RANGE);
      Record.push_back(D->getFirstExprIndex());
      Writer.AddSourceLocation(D->getLBracketLoc(), Record);
      Writer.AddSourceLocation(D->getEllipsisLoc(), Record);
      Writer.AddSourceLocation(D->getRBracketLoc(), Record);
    }
  }
  Code = serialization::EXPR_DESIGNATED_INIT;
}

void ASTStmtWriter::VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getBase());
  Writer.AddStmt(E->getUpdater());
  Code = serialization::EXPR_DESIGNATED_INIT_UPDATE;
}

void ASTStmtWriter::VisitNoInitExpr(NoInitExpr *E) {
  VisitExpr(E);
  Code = serialization::EXPR_NO_INIT;
}

void ASTStmtWriter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  VisitExpr(E);
  Code = serialization::EXPR_IMPLICIT_VALUE_INIT;
}

void ASTStmtWriter::VisitVAArgExpr(VAArgExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Writer.AddTypeSourceInfo(E->getWrittenTypeInfo(), Record);
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Record.push_back(E->isMicrosoftABI());
  Code = serialization::EXPR_VA_ARG;
}

void ASTStmtWriter::VisitAddrLabelExpr(AddrLabelExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getAmpAmpLoc(), Record);
  Writer.AddSourceLocation(E->getLabelLoc(), Record);
  Writer.AddDeclRef(E->getLabel(), Record);
  Code = serialization::EXPR_ADDR_LABEL;
}

void ASTStmtWriter::VisitStmtExpr(StmtExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubStmt());
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_STMT;
}

void ASTStmtWriter::VisitChooseExpr(ChooseExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getCond());
  Writer.AddStmt(E->getLHS());
  Writer.AddStmt(E->getRHS());
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Record.push_back(E->isConditionDependent() ? false : E->isConditionTrue());
  Code = serialization::EXPR_CHOOSE;
}

void ASTStmtWriter::VisitGNUNullExpr(GNUNullExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getTokenLocation(), Record);
  Code = serialization::EXPR_GNU_NULL;
}

void ASTStmtWriter::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumSubExprs());
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I)
    Writer.AddStmt(E->getExpr(I));
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_SHUFFLE_VECTOR;
}

void ASTStmtWriter::VisitConvertVectorExpr(ConvertVectorExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Writer.AddTypeSourceInfo(E->getTypeSourceInfo(), Record);
  Writer.AddStmt(E->getSrcExpr());
  Code = serialization::EXPR_CONVERT_VECTOR;
}

void ASTStmtWriter::VisitBlockExpr(BlockExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getBlockDecl(), Record);
  Code = serialization::EXPR_BLOCK;
}

void ASTStmtWriter::VisitGenericSelectionExpr(GenericSelectionExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumAssocs());

  Writer.AddStmt(E->getControllingExpr());
  for (unsigned I = 0, N = E->getNumAssocs(); I != N; ++I) {
    Writer.AddTypeSourceInfo(E->getAssocTypeSourceInfo(I), Record);
    Writer.AddStmt(E->getAssocExpr(I));
  }
  Record.push_back(E->isResultDependent() ? -1U : E->getResultIndex());

  Writer.AddSourceLocation(E->getGenericLoc(), Record);
  Writer.AddSourceLocation(E->getDefaultLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_GENERIC_SELECTION;
}

void ASTStmtWriter::VisitPseudoObjectExpr(PseudoObjectExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumSemanticExprs());

  // Push the result index.  Currently, this needs to exactly match
  // the encoding used internally for ResultIndex.
  unsigned result = E->getResultExprIndex();
  result = (result == PseudoObjectExpr::NoResult ? 0 : result + 1);
  Record.push_back(result);

  Writer.AddStmt(E->getSyntacticForm());
  for (PseudoObjectExpr::semantics_iterator
         i = E->semantics_begin(), e = E->semantics_end(); i != e; ++i) {
    Writer.AddStmt(*i);
  }
  Code = serialization::EXPR_PSEUDO_OBJECT;
}

void ASTStmtWriter::VisitAtomicExpr(AtomicExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getOp());
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I)
    Writer.AddStmt(E->getSubExprs()[I]);
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_ATOMIC;
}

//===----------------------------------------------------------------------===//
// Objective-C Expressions and Statements.
//===----------------------------------------------------------------------===//

void ASTStmtWriter::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getString());
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Code = serialization::EXPR_OBJC_STRING_LITERAL;
}

void ASTStmtWriter::VisitObjCBoxedExpr(ObjCBoxedExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Writer.AddDeclRef(E->getBoxingMethod(), Record);
  Writer.AddSourceRange(E->getSourceRange(), Record);
  Code = serialization::EXPR_OBJC_BOXED_EXPRESSION;
}

void ASTStmtWriter::VisitObjCArrayLiteral(ObjCArrayLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getNumElements());
  for (unsigned i = 0; i < E->getNumElements(); i++)
    Writer.AddStmt(E->getElement(i));
  Writer.AddDeclRef(E->getArrayWithObjectsMethod(), Record);
  Writer.AddSourceRange(E->getSourceRange(), Record);
  Code = serialization::EXPR_OBJC_ARRAY_LITERAL;
}

void ASTStmtWriter::VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getNumElements());
  Record.push_back(E->HasPackExpansions);
  for (unsigned i = 0; i < E->getNumElements(); i++) {
    ObjCDictionaryElement Element = E->getKeyValueElement(i);
    Writer.AddStmt(Element.Key);
    Writer.AddStmt(Element.Value);
    if (E->HasPackExpansions) {
      Writer.AddSourceLocation(Element.EllipsisLoc, Record);
      unsigned NumExpansions = 0;
      if (Element.NumExpansions)
        NumExpansions = *Element.NumExpansions + 1;
      Record.push_back(NumExpansions);
    }
  }
    
  Writer.AddDeclRef(E->getDictWithObjectsMethod(), Record);
  Writer.AddSourceRange(E->getSourceRange(), Record);
  Code = serialization::EXPR_OBJC_DICTIONARY_LITERAL;
}

void ASTStmtWriter::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  VisitExpr(E);
  Writer.AddTypeSourceInfo(E->getEncodedTypeSourceInfo(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_OBJC_ENCODE;
}

void ASTStmtWriter::VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
  VisitExpr(E);
  Writer.AddSelectorRef(E->getSelector(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_OBJC_SELECTOR_EXPR;
}

void ASTStmtWriter::VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getProtocol(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->ProtoLoc, Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_OBJC_PROTOCOL_EXPR;
}

void ASTStmtWriter::VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getDecl(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddSourceLocation(E->getOpLoc(), Record);
  Writer.AddStmt(E->getBase());
  Record.push_back(E->isArrow());
  Record.push_back(E->isFreeIvar());
  Code = serialization::EXPR_OBJC_IVAR_REF_EXPR;
}

void ASTStmtWriter::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  VisitExpr(E);
  Record.push_back(E->SetterAndMethodRefFlags.getInt());
  Record.push_back(E->isImplicitProperty());
  if (E->isImplicitProperty()) {
    Writer.AddDeclRef(E->getImplicitPropertyGetter(), Record);
    Writer.AddDeclRef(E->getImplicitPropertySetter(), Record);
  } else {
    Writer.AddDeclRef(E->getExplicitProperty(), Record);
  }
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddSourceLocation(E->getReceiverLocation(), Record);
  if (E->isObjectReceiver()) {
    Record.push_back(0);
    Writer.AddStmt(E->getBase());
  } else if (E->isSuperReceiver()) {
    Record.push_back(1);
    Writer.AddTypeRef(E->getSuperReceiverType(), Record);
  } else {
    Record.push_back(2);
    Writer.AddDeclRef(E->getClassReceiver(), Record);
  }
  
  Code = serialization::EXPR_OBJC_PROPERTY_REF_EXPR;
}

void ASTStmtWriter::VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getRBracket(), Record);
  Writer.AddStmt(E->getBaseExpr());
  Writer.AddStmt(E->getKeyExpr());
  Writer.AddDeclRef(E->getAtIndexMethodDecl(), Record);
  Writer.AddDeclRef(E->setAtIndexMethodDecl(), Record);
  
  Code = serialization::EXPR_OBJC_SUBSCRIPT_REF_EXPR;
}

void ASTStmtWriter::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumArgs());
  Record.push_back(E->getNumStoredSelLocs());
  Record.push_back(E->SelLocsKind);
  Record.push_back(E->isDelegateInitCall());
  Record.push_back(E->IsImplicit);
  Record.push_back((unsigned)E->getReceiverKind()); // FIXME: stable encoding
  switch (E->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    Writer.AddStmt(E->getInstanceReceiver());
    break;

  case ObjCMessageExpr::Class:
    Writer.AddTypeSourceInfo(E->getClassReceiverTypeInfo(), Record);
    break;

  case ObjCMessageExpr::SuperClass:
  case ObjCMessageExpr::SuperInstance:
    Writer.AddTypeRef(E->getSuperType(), Record);
    Writer.AddSourceLocation(E->getSuperLoc(), Record);
    break;
  }

  if (E->getMethodDecl()) {
    Record.push_back(1);
    Writer.AddDeclRef(E->getMethodDecl(), Record);
  } else {
    Record.push_back(0);
    Writer.AddSelectorRef(E->getSelector(), Record);    
  }
    
  Writer.AddSourceLocation(E->getLeftLoc(), Record);
  Writer.AddSourceLocation(E->getRightLoc(), Record);

  for (CallExpr::arg_iterator Arg = E->arg_begin(), ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg)
    Writer.AddStmt(*Arg);

  SourceLocation *Locs = E->getStoredSelLocs();
  for (unsigned i = 0, e = E->getNumStoredSelLocs(); i != e; ++i)
    Writer.AddSourceLocation(Locs[i], Record);

  Code = serialization::EXPR_OBJC_MESSAGE_EXPR;
}

void ASTStmtWriter::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  VisitStmt(S);
  Writer.AddStmt(S->getElement());
  Writer.AddStmt(S->getCollection());
  Writer.AddStmt(S->getBody());
  Writer.AddSourceLocation(S->getForLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = serialization::STMT_OBJC_FOR_COLLECTION;
}

void ASTStmtWriter::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  Writer.AddStmt(S->getCatchBody());
  Writer.AddDeclRef(S->getCatchParamDecl(), Record);
  Writer.AddSourceLocation(S->getAtCatchLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Code = serialization::STMT_OBJC_CATCH;
}

void ASTStmtWriter::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  Writer.AddStmt(S->getFinallyBody());
  Writer.AddSourceLocation(S->getAtFinallyLoc(), Record);
  Code = serialization::STMT_OBJC_FINALLY;
}

void ASTStmtWriter::VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *S) {
  Writer.AddStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getAtLoc(), Record);
  Code = serialization::STMT_OBJC_AUTORELEASE_POOL;
}

void ASTStmtWriter::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  Record.push_back(S->getNumCatchStmts());
  Record.push_back(S->getFinallyStmt() != nullptr);
  Writer.AddStmt(S->getTryBody());
  for (unsigned I = 0, N = S->getNumCatchStmts(); I != N; ++I)
    Writer.AddStmt(S->getCatchStmt(I));
  if (S->getFinallyStmt())
    Writer.AddStmt(S->getFinallyStmt());
  Writer.AddSourceLocation(S->getAtTryLoc(), Record);
  Code = serialization::STMT_OBJC_AT_TRY;
}

void ASTStmtWriter::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
  Writer.AddStmt(S->getSynchExpr());
  Writer.AddStmt(S->getSynchBody());
  Writer.AddSourceLocation(S->getAtSynchronizedLoc(), Record);
  Code = serialization::STMT_OBJC_AT_SYNCHRONIZED;
}

void ASTStmtWriter::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  Writer.AddStmt(S->getThrowExpr());
  Writer.AddSourceLocation(S->getThrowLoc(), Record);
  Code = serialization::STMT_OBJC_AT_THROW;
}

void ASTStmtWriter::VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getValue());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = serialization::EXPR_OBJC_BOOL_LITERAL;
}

//===----------------------------------------------------------------------===//
// C++ Expressions and Statements.
//===----------------------------------------------------------------------===//

void ASTStmtWriter::VisitCXXCatchStmt(CXXCatchStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getCatchLoc(), Record);
  Writer.AddDeclRef(S->getExceptionDecl(), Record);
  Writer.AddStmt(S->getHandlerBlock());
  Code = serialization::STMT_CXX_CATCH;
}

void ASTStmtWriter::VisitCXXTryStmt(CXXTryStmt *S) {
  VisitStmt(S);
  Record.push_back(S->getNumHandlers());
  Writer.AddSourceLocation(S->getTryLoc(), Record);
  Writer.AddStmt(S->getTryBlock());
  for (unsigned i = 0, e = S->getNumHandlers(); i != e; ++i)
    Writer.AddStmt(S->getHandler(i));
  Code = serialization::STMT_CXX_TRY;
}

void ASTStmtWriter::VisitCXXForRangeStmt(CXXForRangeStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getForLoc(), Record);
  Writer.AddSourceLocation(S->getCoawaitLoc(), Record);
  Writer.AddSourceLocation(S->getColonLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Writer.AddStmt(S->getRangeStmt());
  Writer.AddStmt(S->getBeginEndStmt());
  Writer.AddStmt(S->getCond());
  Writer.AddStmt(S->getInc());
  Writer.AddStmt(S->getLoopVarStmt());
  Writer.AddStmt(S->getBody());
  Code = serialization::STMT_CXX_FOR_RANGE;
}

void ASTStmtWriter::VisitMSDependentExistsStmt(MSDependentExistsStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getKeywordLoc(), Record);
  Record.push_back(S->isIfExists());
  Writer.AddNestedNameSpecifierLoc(S->getQualifierLoc(), Record);
  Writer.AddDeclarationNameInfo(S->getNameInfo(), Record);
  Writer.AddStmt(S->getSubStmt());
  Code = serialization::STMT_MS_DEPENDENT_EXISTS;
}

void ASTStmtWriter::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  VisitCallExpr(E);
  Record.push_back(E->getOperator());
  Writer.AddSourceRange(E->Range, Record);
  Record.push_back(E->isFPContractable());
  Code = serialization::EXPR_CXX_OPERATOR_CALL;
}

void ASTStmtWriter::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  VisitCallExpr(E);
  Code = serialization::EXPR_CXX_MEMBER_CALL;
}

void ASTStmtWriter::VisitCXXConstructExpr(CXXConstructExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumArgs());
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    Writer.AddStmt(E->getArg(I));
  Writer.AddDeclRef(E->getConstructor(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->isElidable());
  Record.push_back(E->hadMultipleCandidates());
  Record.push_back(E->isListInitialization());
  Record.push_back(E->isStdInitListInitialization());
  Record.push_back(E->requiresZeroInitialization());
  Record.push_back(E->getConstructionKind()); // FIXME: stable encoding
  Writer.AddSourceRange(E->getParenOrBraceRange(), Record);
  Code = serialization::EXPR_CXX_CONSTRUCT;
}

void ASTStmtWriter::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E) {
  VisitCXXConstructExpr(E);
  Writer.AddTypeSourceInfo(E->getTypeSourceInfo(), Record);
  Code = serialization::EXPR_CXX_TEMPORARY_OBJECT;
}

void ASTStmtWriter::VisitLambdaExpr(LambdaExpr *E) {
  VisitExpr(E);
  Record.push_back(E->NumCaptures);
  unsigned NumArrayIndexVars = 0;
  if (E->HasArrayIndexVars)
    NumArrayIndexVars = E->getArrayIndexStarts()[E->NumCaptures];
  Record.push_back(NumArrayIndexVars);
  Writer.AddSourceRange(E->IntroducerRange, Record);
  Record.push_back(E->CaptureDefault); // FIXME: stable encoding
  Writer.AddSourceLocation(E->CaptureDefaultLoc, Record);
  Record.push_back(E->ExplicitParams);
  Record.push_back(E->ExplicitResultType);
  Writer.AddSourceLocation(E->ClosingBrace, Record);
  
  // Add capture initializers.
  for (LambdaExpr::capture_init_iterator C = E->capture_init_begin(),
                                      CEnd = E->capture_init_end();
       C != CEnd; ++C) {
    Writer.AddStmt(*C);
  }
  
  // Add array index variables, if any.
  if (NumArrayIndexVars) {
    Record.append(E->getArrayIndexStarts(), 
                  E->getArrayIndexStarts() + E->NumCaptures + 1);
    VarDecl **ArrayIndexVars = E->getArrayIndexVars();
    for (unsigned I = 0; I != NumArrayIndexVars; ++I)
      Writer.AddDeclRef(ArrayIndexVars[I], Record);
  }
  
  Code = serialization::EXPR_LAMBDA;
}

void ASTStmtWriter::VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_CXX_STD_INITIALIZER_LIST;
}

void ASTStmtWriter::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  VisitExplicitCastExpr(E);
  Writer.AddSourceRange(SourceRange(E->getOperatorLoc(), E->getRParenLoc()),
                        Record);
  Writer.AddSourceRange(E->getAngleBrackets(), Record);
}

void ASTStmtWriter::VisitCXXStaticCastExpr(CXXStaticCastExpr *E) {
  VisitCXXNamedCastExpr(E);
  Code = serialization::EXPR_CXX_STATIC_CAST;
}

void ASTStmtWriter::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E) {
  VisitCXXNamedCastExpr(E);
  Code = serialization::EXPR_CXX_DYNAMIC_CAST;
}

void ASTStmtWriter::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E) {
  VisitCXXNamedCastExpr(E);
  Code = serialization::EXPR_CXX_REINTERPRET_CAST;
}

void ASTStmtWriter::VisitCXXConstCastExpr(CXXConstCastExpr *E) {
  VisitCXXNamedCastExpr(E);
  Code = serialization::EXPR_CXX_CONST_CAST;
}

void ASTStmtWriter::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E) {
  VisitExplicitCastExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CXX_FUNCTIONAL_CAST;
}

void ASTStmtWriter::VisitUserDefinedLiteral(UserDefinedLiteral *E) {
  VisitCallExpr(E);
  Writer.AddSourceLocation(E->UDSuffixLoc, Record);
  Code = serialization::EXPR_USER_DEFINED_LITERAL;
}

void ASTStmtWriter::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getValue());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = serialization::EXPR_CXX_BOOL_LITERAL;
}

void ASTStmtWriter::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = serialization::EXPR_CXX_NULL_PTR_LITERAL;
}

void ASTStmtWriter::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  VisitExpr(E);
  Writer.AddSourceRange(E->getSourceRange(), Record);
  if (E->isTypeOperand()) {
    Writer.AddTypeSourceInfo(E->getTypeOperandSourceInfo(), Record);
    Code = serialization::EXPR_CXX_TYPEID_TYPE;
  } else {
    Writer.AddStmt(E->getExprOperand());
    Code = serialization::EXPR_CXX_TYPEID_EXPR;
  }
}

void ASTStmtWriter::VisitCXXThisExpr(CXXThisExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->isImplicit());
  Code = serialization::EXPR_CXX_THIS;
}

void ASTStmtWriter::VisitCXXThrowExpr(CXXThrowExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getThrowLoc(), Record);
  Writer.AddStmt(E->getSubExpr());
  Record.push_back(E->isThrownVariableInScope());
  Code = serialization::EXPR_CXX_THROW;
}

void ASTStmtWriter::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getParam(), Record);
  Writer.AddSourceLocation(E->getUsedLocation(), Record);
  Code = serialization::EXPR_CXX_DEFAULT_ARG;
}

void ASTStmtWriter::VisitCXXDefaultInitExpr(CXXDefaultInitExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getField(), Record);
  Writer.AddSourceLocation(E->getExprLoc(), Record);
  Code = serialization::EXPR_CXX_DEFAULT_INIT;
}

void ASTStmtWriter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  VisitExpr(E);
  Writer.AddCXXTemporary(E->getTemporary(), Record);
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_CXX_BIND_TEMPORARY;
}

void ASTStmtWriter::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  VisitExpr(E);
  Writer.AddTypeSourceInfo(E->getTypeSourceInfo(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CXX_SCALAR_VALUE_INIT;
}

void ASTStmtWriter::VisitCXXNewExpr(CXXNewExpr *E) {
  VisitExpr(E);
  Record.push_back(E->isGlobalNew());
  Record.push_back(E->isArray());
  Record.push_back(E->doesUsualArrayDeleteWantSize());
  Record.push_back(E->getNumPlacementArgs());
  Record.push_back(E->StoredInitializationStyle);
  Writer.AddDeclRef(E->getOperatorNew(), Record);
  Writer.AddDeclRef(E->getOperatorDelete(), Record);
  Writer.AddTypeSourceInfo(E->getAllocatedTypeSourceInfo(), Record);
  Writer.AddSourceRange(E->getTypeIdParens(), Record);
  Writer.AddSourceRange(E->getSourceRange(), Record);
  Writer.AddSourceRange(E->getDirectInitRange(), Record);
  for (CXXNewExpr::arg_iterator I = E->raw_arg_begin(), e = E->raw_arg_end();
       I != e; ++I)
    Writer.AddStmt(*I);

  Code = serialization::EXPR_CXX_NEW;
}

void ASTStmtWriter::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  VisitExpr(E);
  Record.push_back(E->isGlobalDelete());
  Record.push_back(E->isArrayForm());
  Record.push_back(E->isArrayFormAsWritten());
  Record.push_back(E->doesUsualArrayDeleteWantSize());
  Writer.AddDeclRef(E->getOperatorDelete(), Record);
  Writer.AddStmt(E->getArgument());
  Writer.AddSourceLocation(E->getSourceRange().getBegin(), Record);
  
  Code = serialization::EXPR_CXX_DELETE;
}

void ASTStmtWriter::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  VisitExpr(E);

  Writer.AddStmt(E->getBase());
  Record.push_back(E->isArrow());
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddNestedNameSpecifierLoc(E->getQualifierLoc(), Record);
  Writer.AddTypeSourceInfo(E->getScopeTypeInfo(), Record);
  Writer.AddSourceLocation(E->getColonColonLoc(), Record);
  Writer.AddSourceLocation(E->getTildeLoc(), Record);

  // PseudoDestructorTypeStorage.
  Writer.AddIdentifierRef(E->getDestroyedTypeIdentifier(), Record);
  if (E->getDestroyedTypeIdentifier())
    Writer.AddSourceLocation(E->getDestroyedTypeLoc(), Record);
  else
    Writer.AddTypeSourceInfo(E->getDestroyedTypeInfo(), Record);

  Code = serialization::EXPR_CXX_PSEUDO_DESTRUCTOR;
}

void ASTStmtWriter::VisitExprWithCleanups(ExprWithCleanups *E) {
  VisitExpr(E);
  Record.push_back(E->getNumObjects());
  for (unsigned i = 0, e = E->getNumObjects(); i != e; ++i)
    Writer.AddDeclRef(E->getObject(i), Record);
  
  Writer.AddStmt(E->getSubExpr());
  Code = serialization::EXPR_EXPR_WITH_CLEANUPS;
}

void
ASTStmtWriter::VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E){
  VisitExpr(E);

  // Don't emit anything here, HasTemplateKWAndArgsInfo must be
  // emitted first.

  Record.push_back(E->HasTemplateKWAndArgsInfo);
  if (E->HasTemplateKWAndArgsInfo) {
    const ASTTemplateKWAndArgsInfo &ArgInfo =
        *E->getTrailingObjects<ASTTemplateKWAndArgsInfo>();
    Record.push_back(ArgInfo.NumTemplateArgs);
    AddTemplateKWAndArgsInfo(ArgInfo,
                             E->getTrailingObjects<TemplateArgumentLoc>());
  }

  if (!E->isImplicitAccess())
    Writer.AddStmt(E->getBase());
  else
    Writer.AddStmt(nullptr);
  Writer.AddTypeRef(E->getBaseType(), Record);
  Record.push_back(E->isArrow());
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddNestedNameSpecifierLoc(E->getQualifierLoc(), Record);
  Writer.AddDeclRef(E->getFirstQualifierFoundInScope(), Record);
  Writer.AddDeclarationNameInfo(E->MemberNameInfo, Record);
  Code = serialization::EXPR_CXX_DEPENDENT_SCOPE_MEMBER;
}

void
ASTStmtWriter::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) {
  VisitExpr(E);

  // Don't emit anything here, HasTemplateKWAndArgsInfo must be
  // emitted first.

  Record.push_back(E->HasTemplateKWAndArgsInfo);
  if (E->HasTemplateKWAndArgsInfo) {
    const ASTTemplateKWAndArgsInfo &ArgInfo =
        *E->getTrailingObjects<ASTTemplateKWAndArgsInfo>();
    Record.push_back(ArgInfo.NumTemplateArgs);
    AddTemplateKWAndArgsInfo(ArgInfo,
                             E->getTrailingObjects<TemplateArgumentLoc>());
  }

  Writer.AddNestedNameSpecifierLoc(E->getQualifierLoc(), Record);
  Writer.AddDeclarationNameInfo(E->NameInfo, Record);
  Code = serialization::EXPR_CXX_DEPENDENT_SCOPE_DECL_REF;
}

void
ASTStmtWriter::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E) {
  VisitExpr(E);
  Record.push_back(E->arg_size());
  for (CXXUnresolvedConstructExpr::arg_iterator
         ArgI = E->arg_begin(), ArgE = E->arg_end(); ArgI != ArgE; ++ArgI)
    Writer.AddStmt(*ArgI);
  Writer.AddTypeSourceInfo(E->getTypeSourceInfo(), Record);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = serialization::EXPR_CXX_UNRESOLVED_CONSTRUCT;
}

void ASTStmtWriter::VisitOverloadExpr(OverloadExpr *E) {
  VisitExpr(E);

  // Don't emit anything here, HasTemplateKWAndArgsInfo must be
  // emitted first.

  Record.push_back(E->HasTemplateKWAndArgsInfo);
  if (E->HasTemplateKWAndArgsInfo) {
    const ASTTemplateKWAndArgsInfo &ArgInfo =
        *E->getTrailingASTTemplateKWAndArgsInfo();
    Record.push_back(ArgInfo.NumTemplateArgs);
    AddTemplateKWAndArgsInfo(ArgInfo, E->getTrailingTemplateArgumentLoc());
  }

  Record.push_back(E->getNumDecls());
  for (OverloadExpr::decls_iterator
         OvI = E->decls_begin(), OvE = E->decls_end(); OvI != OvE; ++OvI) {
    Writer.AddDeclRef(OvI.getDecl(), Record);
    Record.push_back(OvI.getAccess());
  }

  Writer.AddDeclarationNameInfo(E->NameInfo, Record);
  Writer.AddNestedNameSpecifierLoc(E->getQualifierLoc(), Record);
}

void ASTStmtWriter::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E) {
  VisitOverloadExpr(E);
  Record.push_back(E->isArrow());
  Record.push_back(E->hasUnresolvedUsing());
  Writer.AddStmt(!E->isImplicitAccess() ? E->getBase() : nullptr);
  Writer.AddTypeRef(E->getBaseType(), Record);
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Code = serialization::EXPR_CXX_UNRESOLVED_MEMBER;
}

void ASTStmtWriter::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E) {
  VisitOverloadExpr(E);
  Record.push_back(E->requiresADL());
  Record.push_back(E->isOverloaded());
  Writer.AddDeclRef(E->getNamingClass(), Record);
  Code = serialization::EXPR_CXX_UNRESOLVED_LOOKUP;
}

void ASTStmtWriter::VisitTypeTraitExpr(TypeTraitExpr *E) {
  VisitExpr(E);
  Record.push_back(E->TypeTraitExprBits.NumArgs);
  Record.push_back(E->TypeTraitExprBits.Kind); // FIXME: Stable encoding
  Record.push_back(E->TypeTraitExprBits.Value);
  Writer.AddSourceRange(E->getSourceRange(), Record);
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    Writer.AddTypeSourceInfo(E->getArg(I), Record);
  Code = serialization::EXPR_TYPE_TRAIT;
}

void ASTStmtWriter::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getTrait());
  Record.push_back(E->getValue());
  Writer.AddSourceRange(E->getSourceRange(), Record);
  Writer.AddTypeSourceInfo(E->getQueriedTypeSourceInfo(), Record);
  Code = serialization::EXPR_ARRAY_TYPE_TRAIT;
}

void ASTStmtWriter::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getTrait());
  Record.push_back(E->getValue());
  Writer.AddSourceRange(E->getSourceRange(), Record);
  Writer.AddStmt(E->getQueriedExpression());
  Code = serialization::EXPR_CXX_EXPRESSION_TRAIT;
}

void ASTStmtWriter::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getValue());
  Writer.AddSourceRange(E->getSourceRange(), Record);
  Writer.AddStmt(E->getOperand());
  Code = serialization::EXPR_CXX_NOEXCEPT;
}

void ASTStmtWriter::VisitPackExpansionExpr(PackExpansionExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getEllipsisLoc(), Record);
  Record.push_back(E->NumExpansions);
  Writer.AddStmt(E->getPattern());
  Code = serialization::EXPR_PACK_EXPANSION;
}

void ASTStmtWriter::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  VisitExpr(E);
  Record.push_back(E->isPartiallySubstituted() ? E->getPartialArguments().size()
                                               : 0);
  Writer.AddSourceLocation(E->OperatorLoc, Record);
  Writer.AddSourceLocation(E->PackLoc, Record);
  Writer.AddSourceLocation(E->RParenLoc, Record);
  Writer.AddDeclRef(E->Pack, Record);
  if (E->isPartiallySubstituted()) {
    for (const auto &TA : E->getPartialArguments())
      Writer.AddTemplateArgument(TA, Record);
  } else if (!E->isValueDependent()) {
    Record.push_back(E->getPackLength());
  }
  Code = serialization::EXPR_SIZEOF_PACK;
}

void ASTStmtWriter::VisitSubstNonTypeTemplateParmExpr(
                                              SubstNonTypeTemplateParmExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getParameter(), Record);
  Writer.AddSourceLocation(E->getNameLoc(), Record);
  Writer.AddStmt(E->getReplacement());
  Code = serialization::EXPR_SUBST_NON_TYPE_TEMPLATE_PARM;
}

void ASTStmtWriter::VisitSubstNonTypeTemplateParmPackExpr(
                                          SubstNonTypeTemplateParmPackExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getParameterPack(), Record);
  Writer.AddTemplateArgument(E->getArgumentPack(), Record);
  Writer.AddSourceLocation(E->getParameterPackLocation(), Record);
  Code = serialization::EXPR_SUBST_NON_TYPE_TEMPLATE_PARM_PACK;
}

void ASTStmtWriter::VisitFunctionParmPackExpr(FunctionParmPackExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumExpansions());
  Writer.AddDeclRef(E->getParameterPack(), Record);
  Writer.AddSourceLocation(E->getParameterPackLocation(), Record);
  for (FunctionParmPackExpr::iterator I = E->begin(), End = E->end();
       I != End; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = serialization::EXPR_FUNCTION_PARM_PACK;
}

void ASTStmtWriter::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getTemporary());
  Writer.AddDeclRef(E->getExtendingDecl(), Record);
  Record.push_back(E->getManglingNumber());
  Code = serialization::EXPR_MATERIALIZE_TEMPORARY;
}

void ASTStmtWriter::VisitCXXFoldExpr(CXXFoldExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->LParenLoc, Record);
  Writer.AddSourceLocation(E->EllipsisLoc, Record);
  Writer.AddSourceLocation(E->RParenLoc, Record);
  Writer.AddStmt(E->SubExprs[0]);
  Writer.AddStmt(E->SubExprs[1]);
  Record.push_back(E->Opcode);
  Code = serialization::EXPR_CXX_FOLD;
}

void ASTStmtWriter::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getSourceExpr());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = serialization::EXPR_OPAQUE_VALUE;
}

void ASTStmtWriter::VisitTypoExpr(TypoExpr *E) {
  VisitExpr(E);
  // TODO: Figure out sane writer behavior for a TypoExpr, if necessary
  llvm_unreachable("Cannot write TypoExpr nodes");
}

//===----------------------------------------------------------------------===//
// CUDA Expressions and Statements.
//===----------------------------------------------------------------------===//

void ASTStmtWriter::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *E) {
  VisitCallExpr(E);
  Writer.AddStmt(E->getConfig());
  Code = serialization::EXPR_CUDA_KERNEL_CALL;
}

//===----------------------------------------------------------------------===//
// OpenCL Expressions and Statements.
//===----------------------------------------------------------------------===//
void ASTStmtWriter::VisitAsTypeExpr(AsTypeExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Writer.AddStmt(E->getSrcExpr());
  Code = serialization::EXPR_ASTYPE;
}

//===----------------------------------------------------------------------===//
// Microsoft Expressions and Statements.
//===----------------------------------------------------------------------===//
void ASTStmtWriter::VisitMSPropertyRefExpr(MSPropertyRefExpr *E) {
  VisitExpr(E);
  Record.push_back(E->isArrow());
  Writer.AddStmt(E->getBaseExpr());
  Writer.AddNestedNameSpecifierLoc(E->getQualifierLoc(), Record);
  Writer.AddSourceLocation(E->getMemberLoc(), Record);
  Writer.AddDeclRef(E->getPropertyDecl(), Record);
  Code = serialization::EXPR_CXX_PROPERTY_REF_EXPR;
}

void ASTStmtWriter::VisitMSPropertySubscriptExpr(MSPropertySubscriptExpr *E) {
  VisitExpr(E);
  Writer.AddStmt(E->getBase());
  Writer.AddStmt(E->getIdx());
  Writer.AddSourceLocation(E->getRBracketLoc(), Record);
  Code = serialization::EXPR_CXX_PROPERTY_SUBSCRIPT_EXPR;
}

void ASTStmtWriter::VisitCXXUuidofExpr(CXXUuidofExpr *E) {
  VisitExpr(E);
  Writer.AddSourceRange(E->getSourceRange(), Record);
  if (E->isTypeOperand()) {
    Writer.AddTypeSourceInfo(E->getTypeOperandSourceInfo(), Record);
    Code = serialization::EXPR_CXX_UUIDOF_TYPE;
  } else {
    Writer.AddStmt(E->getExprOperand());
    Code = serialization::EXPR_CXX_UUIDOF_EXPR;
  }
}

void ASTStmtWriter::VisitSEHExceptStmt(SEHExceptStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getExceptLoc(), Record);
  Writer.AddStmt(S->getFilterExpr());
  Writer.AddStmt(S->getBlock());
  Code = serialization::STMT_SEH_EXCEPT;
}

void ASTStmtWriter::VisitSEHFinallyStmt(SEHFinallyStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getFinallyLoc(), Record);
  Writer.AddStmt(S->getBlock());
  Code = serialization::STMT_SEH_FINALLY;
}

void ASTStmtWriter::VisitSEHTryStmt(SEHTryStmt *S) {
  VisitStmt(S);
  Record.push_back(S->getIsCXXTry());
  Writer.AddSourceLocation(S->getTryLoc(), Record);
  Writer.AddStmt(S->getTryBlock());
  Writer.AddStmt(S->getHandler());
  Code = serialization::STMT_SEH_TRY;
}

void ASTStmtWriter::VisitSEHLeaveStmt(SEHLeaveStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getLeaveLoc(), Record);
  Code = serialization::STMT_SEH_LEAVE;
}

//===----------------------------------------------------------------------===//
// OpenMP Clauses.
//===----------------------------------------------------------------------===//

namespace clang {
class OMPClauseWriter : public OMPClauseVisitor<OMPClauseWriter> {
  ASTStmtWriter *Writer;
  ASTWriter::RecordData &Record;
public:
  OMPClauseWriter(ASTStmtWriter *W, ASTWriter::RecordData &Record)
    : Writer(W), Record(Record) { }
#define OPENMP_CLAUSE(Name, Class)    \
  void Visit##Class(Class *S);
#include "clang/Basic/OpenMPKinds.def"
  void writeClause(OMPClause *C);
};
}

void OMPClauseWriter::writeClause(OMPClause *C) {
  Record.push_back(C->getClauseKind());
  Visit(C);
  Writer->Writer.AddSourceLocation(C->getLocStart(), Record);
  Writer->Writer.AddSourceLocation(C->getLocEnd(), Record);
}

void OMPClauseWriter::VisitOMPIfClause(OMPIfClause *C) {
  Record.push_back(C->getNameModifier());
  Writer->Writer.AddSourceLocation(C->getNameModifierLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getColonLoc(), Record);
  Writer->Writer.AddStmt(C->getCondition());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPFinalClause(OMPFinalClause *C) {
  Writer->Writer.AddStmt(C->getCondition());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPNumThreadsClause(OMPNumThreadsClause *C) {
  Writer->Writer.AddStmt(C->getNumThreads());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPSafelenClause(OMPSafelenClause *C) {
  Writer->Writer.AddStmt(C->getSafelen());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPSimdlenClause(OMPSimdlenClause *C) {
  Writer->Writer.AddStmt(C->getSimdlen());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPCollapseClause(OMPCollapseClause *C) {
  Writer->Writer.AddStmt(C->getNumForLoops());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPDefaultClause(OMPDefaultClause *C) {
  Record.push_back(C->getDefaultKind());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getDefaultKindKwLoc(), Record);
}

void OMPClauseWriter::VisitOMPProcBindClause(OMPProcBindClause *C) {
  Record.push_back(C->getProcBindKind());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getProcBindKindKwLoc(), Record);
}

void OMPClauseWriter::VisitOMPScheduleClause(OMPScheduleClause *C) {
  Record.push_back(C->getScheduleKind());
  Record.push_back(C->getFirstScheduleModifier());
  Record.push_back(C->getSecondScheduleModifier());
  Writer->Writer.AddStmt(C->getChunkSize());
  Writer->Writer.AddStmt(C->getHelperChunkSize());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getFirstScheduleModifierLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getSecondScheduleModifierLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getScheduleKindLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getCommaLoc(), Record);
}

void OMPClauseWriter::VisitOMPOrderedClause(OMPOrderedClause *C) {
  Writer->Writer.AddStmt(C->getNumForLoops());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPNowaitClause(OMPNowaitClause *) {}

void OMPClauseWriter::VisitOMPUntiedClause(OMPUntiedClause *) {}

void OMPClauseWriter::VisitOMPMergeableClause(OMPMergeableClause *) {}

void OMPClauseWriter::VisitOMPReadClause(OMPReadClause *) {}

void OMPClauseWriter::VisitOMPWriteClause(OMPWriteClause *) {}

void OMPClauseWriter::VisitOMPUpdateClause(OMPUpdateClause *) {}

void OMPClauseWriter::VisitOMPCaptureClause(OMPCaptureClause *) {}

void OMPClauseWriter::VisitOMPSeqCstClause(OMPSeqCstClause *) {}

void OMPClauseWriter::VisitOMPThreadsClause(OMPThreadsClause *) {}

void OMPClauseWriter::VisitOMPSIMDClause(OMPSIMDClause *) {}

void OMPClauseWriter::VisitOMPNogroupClause(OMPNogroupClause *) {}

void OMPClauseWriter::VisitOMPPrivateClause(OMPPrivateClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  for (auto *VE : C->varlists()) {
    Writer->Writer.AddStmt(VE);
  }
  for (auto *VE : C->private_copies()) {
    Writer->Writer.AddStmt(VE);
  }
}

void OMPClauseWriter::VisitOMPFirstprivateClause(OMPFirstprivateClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  for (auto *VE : C->varlists()) {
    Writer->Writer.AddStmt(VE);
  }
  for (auto *VE : C->private_copies()) {
    Writer->Writer.AddStmt(VE);
  }
  for (auto *VE : C->inits()) {
    Writer->Writer.AddStmt(VE);
  }
}

void OMPClauseWriter::VisitOMPLastprivateClause(OMPLastprivateClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  for (auto *VE : C->varlists())
    Writer->Writer.AddStmt(VE);
  for (auto *E : C->private_copies())
    Writer->Writer.AddStmt(E);
  for (auto *E : C->source_exprs())
    Writer->Writer.AddStmt(E);
  for (auto *E : C->destination_exprs())
    Writer->Writer.AddStmt(E);
  for (auto *E : C->assignment_ops())
    Writer->Writer.AddStmt(E);
}

void OMPClauseWriter::VisitOMPSharedClause(OMPSharedClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  for (auto *VE : C->varlists())
    Writer->Writer.AddStmt(VE);
}

void OMPClauseWriter::VisitOMPReductionClause(OMPReductionClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getColonLoc(), Record);
  Writer->Writer.AddNestedNameSpecifierLoc(C->getQualifierLoc(), Record);
  Writer->Writer.AddDeclarationNameInfo(C->getNameInfo(), Record);
  for (auto *VE : C->varlists())
    Writer->Writer.AddStmt(VE);
  for (auto *VE : C->privates())
    Writer->Writer.AddStmt(VE);
  for (auto *E : C->lhs_exprs())
    Writer->Writer.AddStmt(E);
  for (auto *E : C->rhs_exprs())
    Writer->Writer.AddStmt(E);
  for (auto *E : C->reduction_ops())
    Writer->Writer.AddStmt(E);
}

void OMPClauseWriter::VisitOMPLinearClause(OMPLinearClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getColonLoc(), Record);
  Record.push_back(C->getModifier());
  Writer->Writer.AddSourceLocation(C->getModifierLoc(), Record);
  for (auto *VE : C->varlists()) {
    Writer->Writer.AddStmt(VE);
  }
  for (auto *VE : C->privates()) {
    Writer->Writer.AddStmt(VE);
  }
  for (auto *VE : C->inits()) {
    Writer->Writer.AddStmt(VE);
  }
  for (auto *VE : C->updates()) {
    Writer->Writer.AddStmt(VE);
  }
  for (auto *VE : C->finals()) {
    Writer->Writer.AddStmt(VE);
  }
  Writer->Writer.AddStmt(C->getStep());
  Writer->Writer.AddStmt(C->getCalcStep());
}

void OMPClauseWriter::VisitOMPAlignedClause(OMPAlignedClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getColonLoc(), Record);
  for (auto *VE : C->varlists())
    Writer->Writer.AddStmt(VE);
  Writer->Writer.AddStmt(C->getAlignment());
}

void OMPClauseWriter::VisitOMPCopyinClause(OMPCopyinClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  for (auto *VE : C->varlists())
    Writer->Writer.AddStmt(VE);
  for (auto *E : C->source_exprs())
    Writer->Writer.AddStmt(E);
  for (auto *E : C->destination_exprs())
    Writer->Writer.AddStmt(E);
  for (auto *E : C->assignment_ops())
    Writer->Writer.AddStmt(E);
}

void OMPClauseWriter::VisitOMPCopyprivateClause(OMPCopyprivateClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  for (auto *VE : C->varlists())
    Writer->Writer.AddStmt(VE);
  for (auto *E : C->source_exprs())
    Writer->Writer.AddStmt(E);
  for (auto *E : C->destination_exprs())
    Writer->Writer.AddStmt(E);
  for (auto *E : C->assignment_ops())
    Writer->Writer.AddStmt(E);
}

void OMPClauseWriter::VisitOMPFlushClause(OMPFlushClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  for (auto *VE : C->varlists())
    Writer->Writer.AddStmt(VE);
}

void OMPClauseWriter::VisitOMPDependClause(OMPDependClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Record.push_back(C->getDependencyKind());
  Writer->Writer.AddSourceLocation(C->getDependencyLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getColonLoc(), Record);
  for (auto *VE : C->varlists())
    Writer->Writer.AddStmt(VE);
}

void OMPClauseWriter::VisitOMPDeviceClause(OMPDeviceClause *C) {
  Writer->Writer.AddStmt(C->getDevice());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPMapClause(OMPMapClause *C) {
  Record.push_back(C->varlist_size());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Record.push_back(C->getMapTypeModifier());
  Record.push_back(C->getMapType());
  Writer->Writer.AddSourceLocation(C->getMapLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getColonLoc(), Record);
  for (auto *VE : C->varlists())
    Writer->Writer.AddStmt(VE);
}

void OMPClauseWriter::VisitOMPNumTeamsClause(OMPNumTeamsClause *C) {
  Writer->Writer.AddStmt(C->getNumTeams());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPThreadLimitClause(OMPThreadLimitClause *C) {
  Writer->Writer.AddStmt(C->getThreadLimit());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPPriorityClause(OMPPriorityClause *C) {
  Writer->Writer.AddStmt(C->getPriority());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPGrainsizeClause(OMPGrainsizeClause *C) {
  Writer->Writer.AddStmt(C->getGrainsize());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPNumTasksClause(OMPNumTasksClause *C) {
  Writer->Writer.AddStmt(C->getNumTasks());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPHintClause(OMPHintClause *C) {
  Writer->Writer.AddStmt(C->getHint());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
}

void OMPClauseWriter::VisitOMPDistScheduleClause(OMPDistScheduleClause *C) {
  Record.push_back(C->getDistScheduleKind());
  Writer->Writer.AddStmt(C->getChunkSize());
  Writer->Writer.AddStmt(C->getHelperChunkSize());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getDistScheduleKindLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getCommaLoc(), Record);
}

void OMPClauseWriter::VisitOMPDefaultmapClause(OMPDefaultmapClause *C) {
  Record.push_back(C->getDefaultmapKind());
  Record.push_back(C->getDefaultmapModifier());
  Writer->Writer.AddSourceLocation(C->getLParenLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getDefaultmapModifierLoc(), Record);
  Writer->Writer.AddSourceLocation(C->getDefaultmapKindLoc(), Record);
}

//===----------------------------------------------------------------------===//
// OpenMP Directives.
//===----------------------------------------------------------------------===//
void ASTStmtWriter::VisitOMPExecutableDirective(OMPExecutableDirective *E) {
  Writer.AddSourceLocation(E->getLocStart(), Record);
  Writer.AddSourceLocation(E->getLocEnd(), Record);
  OMPClauseWriter ClauseWriter(this, Record);
  for (unsigned i = 0; i < E->getNumClauses(); ++i) {
    ClauseWriter.writeClause(E->getClause(i));
  }
  if (E->hasAssociatedStmt())
    Writer.AddStmt(E->getAssociatedStmt());
}

void ASTStmtWriter::VisitOMPLoopDirective(OMPLoopDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  Record.push_back(D->getCollapsedNumber());
  VisitOMPExecutableDirective(D);
  Writer.AddStmt(D->getIterationVariable());
  Writer.AddStmt(D->getLastIteration());
  Writer.AddStmt(D->getCalcLastIteration());
  Writer.AddStmt(D->getPreCond());
  Writer.AddStmt(D->getCond());
  Writer.AddStmt(D->getInit());
  Writer.AddStmt(D->getInc());
  if (isOpenMPWorksharingDirective(D->getDirectiveKind())) {
    Writer.AddStmt(D->getIsLastIterVariable());
    Writer.AddStmt(D->getLowerBoundVariable());
    Writer.AddStmt(D->getUpperBoundVariable());
    Writer.AddStmt(D->getStrideVariable());
    Writer.AddStmt(D->getEnsureUpperBound());
    Writer.AddStmt(D->getNextLowerBound());
    Writer.AddStmt(D->getNextUpperBound());
  }
  for (auto I : D->counters()) {
    Writer.AddStmt(I);
  }
  for (auto I : D->private_counters()) {
    Writer.AddStmt(I);
  }
  for (auto I : D->inits()) {
    Writer.AddStmt(I);
  }
  for (auto I : D->updates()) {
    Writer.AddStmt(I);
  }
  for (auto I : D->finals()) {
    Writer.AddStmt(I);
  }
}

void ASTStmtWriter::VisitOMPParallelDirective(OMPParallelDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Record.push_back(D->hasCancel() ? 1 : 0);
  Code = serialization::STMT_OMP_PARALLEL_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPSimdDirective(OMPSimdDirective *D) {
  VisitOMPLoopDirective(D);
  Code = serialization::STMT_OMP_SIMD_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPForDirective(OMPForDirective *D) {
  VisitOMPLoopDirective(D);
  Record.push_back(D->hasCancel() ? 1 : 0);
  Code = serialization::STMT_OMP_FOR_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPForSimdDirective(OMPForSimdDirective *D) {
  VisitOMPLoopDirective(D);
  Code = serialization::STMT_OMP_FOR_SIMD_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPSectionsDirective(OMPSectionsDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Record.push_back(D->hasCancel() ? 1 : 0);
  Code = serialization::STMT_OMP_SECTIONS_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPSectionDirective(OMPSectionDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
  Record.push_back(D->hasCancel() ? 1 : 0);
  Code = serialization::STMT_OMP_SECTION_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPSingleDirective(OMPSingleDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_SINGLE_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPMasterDirective(OMPMasterDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_MASTER_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPCriticalDirective(OMPCriticalDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Writer.AddDeclarationNameInfo(D->getDirectiveName(), Record);
  Code = serialization::STMT_OMP_CRITICAL_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPParallelForDirective(OMPParallelForDirective *D) {
  VisitOMPLoopDirective(D);
  Record.push_back(D->hasCancel() ? 1 : 0);
  Code = serialization::STMT_OMP_PARALLEL_FOR_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPParallelForSimdDirective(
    OMPParallelForSimdDirective *D) {
  VisitOMPLoopDirective(D);
  Code = serialization::STMT_OMP_PARALLEL_FOR_SIMD_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPParallelSectionsDirective(
    OMPParallelSectionsDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Record.push_back(D->hasCancel() ? 1 : 0);
  Code = serialization::STMT_OMP_PARALLEL_SECTIONS_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTaskDirective(OMPTaskDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Record.push_back(D->hasCancel() ? 1 : 0);
  Code = serialization::STMT_OMP_TASK_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPAtomicDirective(OMPAtomicDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Writer.AddStmt(D->getX());
  Writer.AddStmt(D->getV());
  Writer.AddStmt(D->getExpr());
  Writer.AddStmt(D->getUpdateExpr());
  Record.push_back(D->isXLHSInRHSPart() ? 1 : 0);
  Record.push_back(D->isPostfixUpdate() ? 1 : 0);
  Code = serialization::STMT_OMP_ATOMIC_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTargetDirective(OMPTargetDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_TARGET_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTargetDataDirective(OMPTargetDataDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_TARGET_DATA_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTargetEnterDataDirective(
    OMPTargetEnterDataDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_TARGET_ENTER_DATA_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTargetExitDataDirective(
    OMPTargetExitDataDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_TARGET_EXIT_DATA_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTargetParallelDirective(
    OMPTargetParallelDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_TARGET_PARALLEL_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTargetParallelForDirective(
    OMPTargetParallelForDirective *D) {
  VisitOMPLoopDirective(D);
  Record.push_back(D->hasCancel() ? 1 : 0);
  Code = serialization::STMT_OMP_TARGET_PARALLEL_FOR_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTaskyieldDirective(OMPTaskyieldDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_TASKYIELD_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPBarrierDirective(OMPBarrierDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_BARRIER_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTaskwaitDirective(OMPTaskwaitDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_TASKWAIT_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTaskgroupDirective(OMPTaskgroupDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_TASKGROUP_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPFlushDirective(OMPFlushDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_FLUSH_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPOrderedDirective(OMPOrderedDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_ORDERED_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTeamsDirective(OMPTeamsDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Code = serialization::STMT_OMP_TEAMS_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPCancellationPointDirective(
    OMPCancellationPointDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
  Record.push_back(D->getCancelRegion());
  Code = serialization::STMT_OMP_CANCELLATION_POINT_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPCancelDirective(OMPCancelDirective *D) {
  VisitStmt(D);
  Record.push_back(D->getNumClauses());
  VisitOMPExecutableDirective(D);
  Record.push_back(D->getCancelRegion());
  Code = serialization::STMT_OMP_CANCEL_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTaskLoopDirective(OMPTaskLoopDirective *D) {
  VisitOMPLoopDirective(D);
  Code = serialization::STMT_OMP_TASKLOOP_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPTaskLoopSimdDirective(OMPTaskLoopSimdDirective *D) {
  VisitOMPLoopDirective(D);
  Code = serialization::STMT_OMP_TASKLOOP_SIMD_DIRECTIVE;
}

void ASTStmtWriter::VisitOMPDistributeDirective(OMPDistributeDirective *D) {
  VisitOMPLoopDirective(D);
  Code = serialization::STMT_OMP_DISTRIBUTE_DIRECTIVE;
}

//===----------------------------------------------------------------------===//
// ASTWriter Implementation
//===----------------------------------------------------------------------===//

unsigned ASTWriter::RecordSwitchCaseID(SwitchCase *S) {
  assert(SwitchCaseIDs.find(S) == SwitchCaseIDs.end() &&
         "SwitchCase recorded twice");
  unsigned NextID = SwitchCaseIDs.size();
  SwitchCaseIDs[S] = NextID;
  return NextID;
}

unsigned ASTWriter::getSwitchCaseID(SwitchCase *S) {
  assert(SwitchCaseIDs.find(S) != SwitchCaseIDs.end() &&
         "SwitchCase hasn't been seen yet");
  return SwitchCaseIDs[S];
}

void ASTWriter::ClearSwitchCaseIDs() {
  SwitchCaseIDs.clear();
}

/// \brief Write the given substatement or subexpression to the
/// bitstream.
void ASTWriter::WriteSubStmt(Stmt *S,
                             llvm::DenseMap<Stmt *, uint64_t> &SubStmtEntries,
                             llvm::DenseSet<Stmt *> &ParentStmts) {
  RecordData Record;
  ASTStmtWriter Writer(*this, Record);
  ++NumStatements;
  
  if (!S) {
    Stream.EmitRecord(serialization::STMT_NULL_PTR, Record);
    return;
  }

  llvm::DenseMap<Stmt *, uint64_t>::iterator I = SubStmtEntries.find(S);
  if (I != SubStmtEntries.end()) {
    Record.push_back(I->second);
    Stream.EmitRecord(serialization::STMT_REF_PTR, Record);
    return;
  }

#ifndef NDEBUG
  assert(!ParentStmts.count(S) && "There is a Stmt cycle!");

  struct ParentStmtInserterRAII {
    Stmt *S;
    llvm::DenseSet<Stmt *> &ParentStmts;

    ParentStmtInserterRAII(Stmt *S, llvm::DenseSet<Stmt *> &ParentStmts)
      : S(S), ParentStmts(ParentStmts) {
      ParentStmts.insert(S);
    }
    ~ParentStmtInserterRAII() {
      ParentStmts.erase(S);
    }
  };

  ParentStmtInserterRAII ParentStmtInserter(S, ParentStmts);
#endif

  // Redirect ASTWriter::AddStmt to collect sub-stmts.
  SmallVector<Stmt *, 16> SubStmts;
  CollectedStmts = &SubStmts;

  Writer.Code = serialization::STMT_NULL_PTR;
  Writer.AbbrevToUse = 0;
  Writer.Visit(S);
  
#ifndef NDEBUG
  if (Writer.Code == serialization::STMT_NULL_PTR) {
    SourceManager &SrcMgr
      = DeclIDs.begin()->first->getASTContext().getSourceManager();
    S->dump(SrcMgr);
    llvm_unreachable("Unhandled sub-statement writing AST file");
  }
#endif

  // Revert ASTWriter::AddStmt.
  CollectedStmts = &StmtsToEmit;

  // Write the sub-stmts in reverse order, last to first. When reading them back
  // we will read them in correct order by "pop"ing them from the Stmts stack.
  // This simplifies reading and allows to store a variable number of sub-stmts
  // without knowing it in advance.
  while (!SubStmts.empty())
    WriteSubStmt(SubStmts.pop_back_val(), SubStmtEntries, ParentStmts);
  
  Stream.EmitRecord(Writer.Code, Record, Writer.AbbrevToUse);
 
  SubStmtEntries[S] = Stream.GetCurrentBitNo();
}

/// \brief Flush all of the statements that have been added to the
/// queue via AddStmt().
void ASTWriter::FlushStmts() {
  RecordData Record;

  // We expect to be the only consumer of the two temporary statement maps,
  // assert that they are empty.
  assert(SubStmtEntries.empty() && "unexpected entries in sub-stmt map");
  assert(ParentStmts.empty() && "unexpected entries in parent stmt map");

  for (unsigned I = 0, N = StmtsToEmit.size(); I != N; ++I) {
    WriteSubStmt(StmtsToEmit[I], SubStmtEntries, ParentStmts);
    
    assert(N == StmtsToEmit.size() &&
           "Substatement written via AddStmt rather than WriteSubStmt!");

    // Note that we are at the end of a full expression. Any
    // expression records that follow this one are part of a different
    // expression.
    Stream.EmitRecord(serialization::STMT_STOP, Record);

    SubStmtEntries.clear();
    ParentStmts.clear();
  }

  StmtsToEmit.clear();
}
