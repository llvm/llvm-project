//===--- ASTReaderStmt.cpp - Stmt/Expr Deserialization ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Statement/expression deserialization.  This implements the
// ASTReader::ReadStmt method.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTReader.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;
using namespace clang::serialization;

namespace clang {

  class ASTStmtReader : public StmtVisitor<ASTStmtReader> {
    friend class OMPClauseReader;

    ASTRecordReader Record;
    llvm::BitstreamCursor &DeclsCursor;
    unsigned &Idx;

    SourceLocation ReadSourceLocation() {
      return Record.ReadSourceLocation(Idx);
    }

    SourceRange ReadSourceRange() {
      return Record.ReadSourceRange(Idx);
    }

    std::string ReadString() {
      return Record.ReadString(Idx);
    }

    TypeSourceInfo *GetTypeSourceInfo() {
      return Record.GetTypeSourceInfo(Idx);
    }

    Decl *ReadDecl() {
      return Record.ReadDecl(Idx);
    }

    template<typename T>
    T *ReadDeclAs() {
      return Record.ReadDeclAs<T>(Idx);
    }

    void ReadDeclarationNameLoc(DeclarationNameLoc &DNLoc,
                                DeclarationName Name) {
      Record.ReadDeclarationNameLoc(DNLoc, Name, Idx);
    }

    void ReadDeclarationNameInfo(DeclarationNameInfo &NameInfo) {
      Record.ReadDeclarationNameInfo(NameInfo, Idx);
    }

  public:
    ASTStmtReader(ASTReader &Reader, ModuleFile &F,
                  llvm::BitstreamCursor &Cursor,
                  const ASTReader::RecordData &Record, unsigned &Idx)
        : Record(Reader, Record, F), DeclsCursor(Cursor), Idx(Idx) { }

    /// \brief The number of record fields required for the Stmt class
    /// itself.
    static const unsigned NumStmtFields = 0;

    /// \brief The number of record fields required for the Expr class
    /// itself.
    static const unsigned NumExprFields = NumStmtFields + 7;

    /// \brief Read and initialize a ExplicitTemplateArgumentList structure.
    void ReadTemplateKWAndArgsInfo(ASTTemplateKWAndArgsInfo &Args,
                                   TemplateArgumentLoc *ArgsLocArray,
                                   unsigned NumTemplateArgs);
    /// \brief Read and initialize a ExplicitTemplateArgumentList structure.
    void ReadExplicitTemplateArgumentList(ASTTemplateArgumentListInfo &ArgList,
                                          unsigned NumTemplateArgs);

    void VisitStmt(Stmt *S);
#define STMT(Type, Base) \
    void Visit##Type(Type *);
#include "clang/AST/StmtNodes.inc"
  };
}

void ASTStmtReader::ReadTemplateKWAndArgsInfo(ASTTemplateKWAndArgsInfo &Args,
                                              TemplateArgumentLoc *ArgsLocArray,
                                              unsigned NumTemplateArgs) {
  SourceLocation TemplateKWLoc = ReadSourceLocation();
  TemplateArgumentListInfo ArgInfo;
  ArgInfo.setLAngleLoc(ReadSourceLocation());
  ArgInfo.setRAngleLoc(ReadSourceLocation());
  for (unsigned i = 0; i != NumTemplateArgs; ++i)
    ArgInfo.addArgument(Record.ReadTemplateArgumentLoc(Idx));
  Args.initializeFrom(TemplateKWLoc, ArgInfo, ArgsLocArray);
}

void ASTStmtReader::VisitStmt(Stmt *S) {
  assert(Idx == NumStmtFields && "Incorrect statement field count");
}

void ASTStmtReader::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
  S->setSemiLoc(ReadSourceLocation());
  S->HasLeadingEmptyMacro = Record[Idx++];
}

void ASTStmtReader::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
  SmallVector<Stmt *, 16> Stmts;
  unsigned NumStmts = Record[Idx++];
  while (NumStmts--)
    Stmts.push_back(Record.ReadSubStmt());
  S->setStmts(Record.getContext(), Stmts);
  S->LBraceLoc = ReadSourceLocation();
  S->RBraceLoc = ReadSourceLocation();
}

void ASTStmtReader::VisitSwitchCase(SwitchCase *S) {
  VisitStmt(S);
  Record.RecordSwitchCaseID(S, Record[Idx++]);
  S->setKeywordLoc(ReadSourceLocation());
  S->setColonLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitCaseStmt(CaseStmt *S) {
  VisitSwitchCase(S);
  S->setLHS(Record.ReadSubExpr());
  S->setRHS(Record.ReadSubExpr());
  S->setSubStmt(Record.ReadSubStmt());
  S->setEllipsisLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitDefaultStmt(DefaultStmt *S) {
  VisitSwitchCase(S);
  S->setSubStmt(Record.ReadSubStmt());
}

void ASTStmtReader::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  LabelDecl *LD = ReadDeclAs<LabelDecl>();
  LD->setStmt(S);
  S->setDecl(LD);
  S->setSubStmt(Record.ReadSubStmt());
  S->setIdentLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitAttributedStmt(AttributedStmt *S) {
  VisitStmt(S);
  uint64_t NumAttrs = Record[Idx++];
  AttrVec Attrs;
  Record.ReadAttributes(Attrs, Idx);
  (void)NumAttrs;
  assert(NumAttrs == S->NumAttrs);
  assert(NumAttrs == Attrs.size());
  std::copy(Attrs.begin(), Attrs.end(), S->getAttrArrayPtr());
  S->SubStmt = Record.ReadSubStmt();
  S->AttrLoc = ReadSourceLocation();
}

void ASTStmtReader::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
  S->setConstexpr(Record[Idx++]);
  S->setInit(Record.ReadSubStmt());
  S->setConditionVariable(Record.getContext(), ReadDeclAs<VarDecl>());
  S->setCond(Record.ReadSubExpr());
  S->setThen(Record.ReadSubStmt());
  S->setElse(Record.ReadSubStmt());
  S->setIfLoc(ReadSourceLocation());
  S->setElseLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
  S->setInit(Record.ReadSubStmt());
  S->setConditionVariable(Record.getContext(), ReadDeclAs<VarDecl>());
  S->setCond(Record.ReadSubExpr());
  S->setBody(Record.ReadSubStmt());
  S->setSwitchLoc(ReadSourceLocation());
  if (Record[Idx++])
    S->setAllEnumCasesCovered();

  SwitchCase *PrevSC = nullptr;
  for (unsigned N = Record.size(); Idx != N; ++Idx) {
    SwitchCase *SC = Record.getSwitchCaseWithID(Record[Idx]);
    if (PrevSC)
      PrevSC->setNextSwitchCase(SC);
    else
      S->setSwitchCaseList(SC);

    PrevSC = SC;
  }
}

void ASTStmtReader::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
  S->setConditionVariable(Record.getContext(), ReadDeclAs<VarDecl>());

  S->setCond(Record.ReadSubExpr());
  S->setBody(Record.ReadSubStmt());
  S->setWhileLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
  S->setCond(Record.ReadSubExpr());
  S->setBody(Record.ReadSubStmt());
  S->setDoLoc(ReadSourceLocation());
  S->setWhileLoc(ReadSourceLocation());
  S->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
  S->setInit(Record.ReadSubStmt());
  S->setCond(Record.ReadSubExpr());
  S->setConditionVariable(Record.getContext(), ReadDeclAs<VarDecl>());
  S->setInc(Record.ReadSubExpr());
  S->setBody(Record.ReadSubStmt());
  S->setForLoc(ReadSourceLocation());
  S->setLParenLoc(ReadSourceLocation());
  S->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  S->setLabel(ReadDeclAs<LabelDecl>());
  S->setGotoLoc(ReadSourceLocation());
  S->setLabelLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
  S->setGotoLoc(ReadSourceLocation());
  S->setStarLoc(ReadSourceLocation());
  S->setTarget(Record.ReadSubExpr());
}

void ASTStmtReader::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
  S->setContinueLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
  S->setBreakLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
  S->setRetValue(Record.ReadSubExpr());
  S->setReturnLoc(ReadSourceLocation());
  S->setNRVOCandidate(ReadDeclAs<VarDecl>());
}

void ASTStmtReader::VisitDeclStmt(DeclStmt *S) {
  VisitStmt(S);
  S->setStartLoc(ReadSourceLocation());
  S->setEndLoc(ReadSourceLocation());

  if (Idx + 1 == Record.size()) {
    // Single declaration
    S->setDeclGroup(DeclGroupRef(ReadDecl()));
  } else {
    SmallVector<Decl *, 16> Decls;
    Decls.reserve(Record.size() - Idx);
    for (unsigned N = Record.size(); Idx != N; )
      Decls.push_back(ReadDecl());
    S->setDeclGroup(DeclGroupRef(DeclGroup::Create(Record.getContext(),
                                                   Decls.data(),
                                                   Decls.size())));
  }
}

void ASTStmtReader::VisitAsmStmt(AsmStmt *S) {
  VisitStmt(S);
  S->NumOutputs = Record[Idx++];
  S->NumInputs = Record[Idx++];
  S->NumClobbers = Record[Idx++];
  S->setAsmLoc(ReadSourceLocation());
  S->setVolatile(Record[Idx++]);
  S->setSimple(Record[Idx++]);
}

void ASTStmtReader::VisitGCCAsmStmt(GCCAsmStmt *S) {
  VisitAsmStmt(S);
  S->setRParenLoc(ReadSourceLocation());
  S->setAsmString(cast_or_null<StringLiteral>(Record.ReadSubStmt()));

  unsigned NumOutputs = S->getNumOutputs();
  unsigned NumInputs = S->getNumInputs();
  unsigned NumClobbers = S->getNumClobbers();

  // Outputs and inputs
  SmallVector<IdentifierInfo *, 16> Names;
  SmallVector<StringLiteral*, 16> Constraints;
  SmallVector<Stmt*, 16> Exprs;
  for (unsigned I = 0, N = NumOutputs + NumInputs; I != N; ++I) {
    Names.push_back(Record.GetIdentifierInfo(Idx));
    Constraints.push_back(cast_or_null<StringLiteral>(Record.ReadSubStmt()));
    Exprs.push_back(Record.ReadSubStmt());
  }

  // Constraints
  SmallVector<StringLiteral*, 16> Clobbers;
  for (unsigned I = 0; I != NumClobbers; ++I)
    Clobbers.push_back(cast_or_null<StringLiteral>(Record.ReadSubStmt()));

  S->setOutputsAndInputsAndClobbers(Record.getContext(),
                                    Names.data(), Constraints.data(),
                                    Exprs.data(), NumOutputs, NumInputs,
                                    Clobbers.data(), NumClobbers);
}

void ASTStmtReader::VisitMSAsmStmt(MSAsmStmt *S) {
  VisitAsmStmt(S);
  S->LBraceLoc = ReadSourceLocation();
  S->EndLoc = ReadSourceLocation();
  S->NumAsmToks = Record[Idx++];
  std::string AsmStr = ReadString();

  // Read the tokens.
  SmallVector<Token, 16> AsmToks;
  AsmToks.reserve(S->NumAsmToks);
  for (unsigned i = 0, e = S->NumAsmToks; i != e; ++i) {
    AsmToks.push_back(Record.ReadToken(Idx));
  }

  // The calls to reserve() for the FooData vectors are mandatory to
  // prevent dead StringRefs in the Foo vectors.

  // Read the clobbers.
  SmallVector<std::string, 16> ClobbersData;
  SmallVector<StringRef, 16> Clobbers;
  ClobbersData.reserve(S->NumClobbers);
  Clobbers.reserve(S->NumClobbers);
  for (unsigned i = 0, e = S->NumClobbers; i != e; ++i) {
    ClobbersData.push_back(ReadString());
    Clobbers.push_back(ClobbersData.back());
  }

  // Read the operands.
  unsigned NumOperands = S->NumOutputs + S->NumInputs;
  SmallVector<Expr*, 16> Exprs;
  SmallVector<std::string, 16> ConstraintsData;
  SmallVector<StringRef, 16> Constraints;
  Exprs.reserve(NumOperands);
  ConstraintsData.reserve(NumOperands);
  Constraints.reserve(NumOperands);
  for (unsigned i = 0; i != NumOperands; ++i) {
    Exprs.push_back(cast<Expr>(Record.ReadSubStmt()));
    ConstraintsData.push_back(ReadString());
    Constraints.push_back(ConstraintsData.back());
  }

  S->initialize(Record.getContext(), AsmStr, AsmToks,
                Constraints, Exprs, Clobbers);
}

void ASTStmtReader::VisitCoroutineBodyStmt(CoroutineBodyStmt *S) {
  // FIXME: Implement coroutine serialization.
  llvm_unreachable("unimplemented");
}

void ASTStmtReader::VisitCoreturnStmt(CoreturnStmt *S) {
  // FIXME: Implement coroutine serialization.
  llvm_unreachable("unimplemented");
}

void ASTStmtReader::VisitCoawaitExpr(CoawaitExpr *S) {
  // FIXME: Implement coroutine serialization.
  llvm_unreachable("unimplemented");
}

void ASTStmtReader::VisitCoyieldExpr(CoyieldExpr *S) {
  // FIXME: Implement coroutine serialization.
  llvm_unreachable("unimplemented");
}

void ASTStmtReader::VisitCapturedStmt(CapturedStmt *S) {
  VisitStmt(S);
  ++Idx;
  S->setCapturedDecl(ReadDeclAs<CapturedDecl>());
  S->setCapturedRegionKind(static_cast<CapturedRegionKind>(Record[Idx++]));
  S->setCapturedRecordDecl(ReadDeclAs<RecordDecl>());

  // Capture inits
  for (CapturedStmt::capture_init_iterator I = S->capture_init_begin(),
                                           E = S->capture_init_end();
       I != E; ++I)
    *I = Record.ReadSubExpr();

  // Body
  S->setCapturedStmt(Record.ReadSubStmt());
  S->getCapturedDecl()->setBody(S->getCapturedStmt());

  // Captures
  for (auto &I : S->captures()) {
    I.VarAndKind.setPointer(ReadDeclAs<VarDecl>());
    I.VarAndKind
        .setInt(static_cast<CapturedStmt::VariableCaptureKind>(Record[Idx++]));
    I.Loc = ReadSourceLocation();
  }
}

void ASTStmtReader::VisitExpr(Expr *E) {
  VisitStmt(E);
  E->setType(Record.readType(Idx));
  E->setTypeDependent(Record[Idx++]);
  E->setValueDependent(Record[Idx++]);
  E->setInstantiationDependent(Record[Idx++]);
  E->ExprBits.ContainsUnexpandedParameterPack = Record[Idx++];
  E->setValueKind(static_cast<ExprValueKind>(Record[Idx++]));
  E->setObjectKind(static_cast<ExprObjectKind>(Record[Idx++]));
  assert(Idx == NumExprFields && "Incorrect expression field count");
}

void ASTStmtReader::VisitPredefinedExpr(PredefinedExpr *E) {
  VisitExpr(E);
  E->setLocation(ReadSourceLocation());
  E->Type = (PredefinedExpr::IdentType)Record[Idx++];
  E->FnName = cast_or_null<StringLiteral>(Record.ReadSubExpr());
}

void ASTStmtReader::VisitDeclRefExpr(DeclRefExpr *E) {
  VisitExpr(E);

  E->DeclRefExprBits.HasQualifier = Record[Idx++];
  E->DeclRefExprBits.HasFoundDecl = Record[Idx++];
  E->DeclRefExprBits.HasTemplateKWAndArgsInfo = Record[Idx++];
  E->DeclRefExprBits.HadMultipleCandidates = Record[Idx++];
  E->DeclRefExprBits.RefersToEnclosingVariableOrCapture = Record[Idx++];
  unsigned NumTemplateArgs = 0;
  if (E->hasTemplateKWAndArgsInfo())
    NumTemplateArgs = Record[Idx++];

  if (E->hasQualifier())
    new (E->getTrailingObjects<NestedNameSpecifierLoc>())
        NestedNameSpecifierLoc(Record.ReadNestedNameSpecifierLoc(Idx));

  if (E->hasFoundDecl())
    *E->getTrailingObjects<NamedDecl *>() = ReadDeclAs<NamedDecl>();

  if (E->hasTemplateKWAndArgsInfo())
    ReadTemplateKWAndArgsInfo(
        *E->getTrailingObjects<ASTTemplateKWAndArgsInfo>(),
        E->getTrailingObjects<TemplateArgumentLoc>(), NumTemplateArgs);

  E->setDecl(ReadDeclAs<ValueDecl>());
  E->setLocation(ReadSourceLocation());
  ReadDeclarationNameLoc(E->DNLoc, E->getDecl()->getDeclName());
}

void ASTStmtReader::VisitIntegerLiteral(IntegerLiteral *E) {
  VisitExpr(E);
  E->setLocation(ReadSourceLocation());
  E->setValue(Record.getContext(), Record.ReadAPInt(Idx));
}

void ASTStmtReader::VisitFloatingLiteral(FloatingLiteral *E) {
  VisitExpr(E);
  E->setRawSemantics(static_cast<Stmt::APFloatSemantics>(Record[Idx++]));
  E->setExact(Record[Idx++]);
  E->setValue(Record.getContext(), Record.ReadAPFloat(E->getSemantics(), Idx));
  E->setLocation(ReadSourceLocation());
}

void ASTStmtReader::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  VisitExpr(E);
  E->setSubExpr(Record.ReadSubExpr());
}

void ASTStmtReader::VisitStringLiteral(StringLiteral *E) {
  VisitExpr(E);
  unsigned Len = Record[Idx++];
  assert(Record[Idx] == E->getNumConcatenated() &&
         "Wrong number of concatenated tokens!");
  ++Idx;
  StringLiteral::StringKind kind =
        static_cast<StringLiteral::StringKind>(Record[Idx++]);
  bool isPascal = Record[Idx++];

  // Read string data
  SmallString<16> Str(&Record[Idx], &Record[Idx] + Len);
  E->setString(Record.getContext(), Str, kind, isPascal);
  Idx += Len;

  // Read source locations
  for (unsigned I = 0, N = E->getNumConcatenated(); I != N; ++I)
    E->setStrTokenLoc(I, ReadSourceLocation());
}

void ASTStmtReader::VisitCharacterLiteral(CharacterLiteral *E) {
  VisitExpr(E);
  E->setValue(Record[Idx++]);
  E->setLocation(ReadSourceLocation());
  E->setKind(static_cast<CharacterLiteral::CharacterKind>(Record[Idx++]));
}

void ASTStmtReader::VisitParenExpr(ParenExpr *E) {
  VisitExpr(E);
  E->setLParen(ReadSourceLocation());
  E->setRParen(ReadSourceLocation());
  E->setSubExpr(Record.ReadSubExpr());
}

void ASTStmtReader::VisitParenListExpr(ParenListExpr *E) {
  VisitExpr(E);
  unsigned NumExprs = Record[Idx++];
  E->Exprs = new (Record.getContext()) Stmt*[NumExprs];
  for (unsigned i = 0; i != NumExprs; ++i)
    E->Exprs[i] = Record.ReadSubStmt();
  E->NumExprs = NumExprs;
  E->LParenLoc = ReadSourceLocation();
  E->RParenLoc = ReadSourceLocation();
}

void ASTStmtReader::VisitUnaryOperator(UnaryOperator *E) {
  VisitExpr(E);
  E->setSubExpr(Record.ReadSubExpr());
  E->setOpcode((UnaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitOffsetOfExpr(OffsetOfExpr *E) {
  VisitExpr(E);
  assert(E->getNumComponents() == Record[Idx]);
  ++Idx;
  assert(E->getNumExpressions() == Record[Idx]);
  ++Idx;
  E->setOperatorLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
  E->setTypeSourceInfo(GetTypeSourceInfo());
  for (unsigned I = 0, N = E->getNumComponents(); I != N; ++I) {
    OffsetOfNode::Kind Kind = static_cast<OffsetOfNode::Kind>(Record[Idx++]);
    SourceLocation Start = ReadSourceLocation();
    SourceLocation End = ReadSourceLocation();
    switch (Kind) {
    case OffsetOfNode::Array:
      E->setComponent(I, OffsetOfNode(Start, Record[Idx++], End));
      break;

    case OffsetOfNode::Field:
      E->setComponent(
          I, OffsetOfNode(Start, ReadDeclAs<FieldDecl>(), End));
      break;

    case OffsetOfNode::Identifier:
      E->setComponent(
          I,
          OffsetOfNode(Start, Record.GetIdentifierInfo(Idx), End));
      break;

    case OffsetOfNode::Base: {
      CXXBaseSpecifier *Base = new (Record.getContext()) CXXBaseSpecifier();
      *Base = Record.ReadCXXBaseSpecifier(Idx);
      E->setComponent(I, OffsetOfNode(Base));
      break;
    }
    }
  }

  for (unsigned I = 0, N = E->getNumExpressions(); I != N; ++I)
    E->setIndexExpr(I, Record.ReadSubExpr());
}

void ASTStmtReader::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) {
  VisitExpr(E);
  E->setKind(static_cast<UnaryExprOrTypeTrait>(Record[Idx++]));
  if (Record[Idx] == 0) {
    E->setArgument(Record.ReadSubExpr());
    ++Idx;
  } else {
    E->setArgument(GetTypeSourceInfo());
  }
  E->setOperatorLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  VisitExpr(E);
  E->setLHS(Record.ReadSubExpr());
  E->setRHS(Record.ReadSubExpr());
  E->setRBracketLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitOMPArraySectionExpr(OMPArraySectionExpr *E) {
  VisitExpr(E);
  E->setBase(Record.ReadSubExpr());
  E->setLowerBound(Record.ReadSubExpr());
  E->setLength(Record.ReadSubExpr());
  E->setColonLoc(ReadSourceLocation());
  E->setRBracketLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitCallExpr(CallExpr *E) {
  VisitExpr(E);
  E->setNumArgs(Record.getContext(), Record[Idx++]);
  E->setRParenLoc(ReadSourceLocation());
  E->setCallee(Record.ReadSubExpr());
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, Record.ReadSubExpr());
}

void ASTStmtReader::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  VisitCallExpr(E);
}

void ASTStmtReader::VisitMemberExpr(MemberExpr *E) {
  // Don't call VisitExpr, this is fully initialized at creation.
  assert(E->getStmtClass() == Stmt::MemberExprClass &&
         "It's a subclass, we must advance Idx!");
}

void ASTStmtReader::VisitObjCIsaExpr(ObjCIsaExpr *E) {
  VisitExpr(E);
  E->setBase(Record.ReadSubExpr());
  E->setIsaMemberLoc(ReadSourceLocation());
  E->setOpLoc(ReadSourceLocation());
  E->setArrow(Record[Idx++]);
}

void ASTStmtReader::
VisitObjCIndirectCopyRestoreExpr(ObjCIndirectCopyRestoreExpr *E) {
  VisitExpr(E);
  E->Operand = Record.ReadSubExpr();
  E->setShouldCopy(Record[Idx++]);
}

void ASTStmtReader::VisitObjCBridgedCastExpr(ObjCBridgedCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->LParenLoc = ReadSourceLocation();
  E->BridgeKeywordLoc = ReadSourceLocation();
  E->Kind = Record[Idx++];
}

void ASTStmtReader::VisitCastExpr(CastExpr *E) {
  VisitExpr(E);
  unsigned NumBaseSpecs = Record[Idx++];
  assert(NumBaseSpecs == E->path_size());
  E->setSubExpr(Record.ReadSubExpr());
  E->setCastKind((CastKind)Record[Idx++]);
  CastExpr::path_iterator BaseI = E->path_begin();
  while (NumBaseSpecs--) {
    CXXBaseSpecifier *BaseSpec = new (Record.getContext()) CXXBaseSpecifier;
    *BaseSpec = Record.ReadCXXBaseSpecifier(Idx);
    *BaseI++ = BaseSpec;
  }
}

void ASTStmtReader::VisitBinaryOperator(BinaryOperator *E) {
  VisitExpr(E);
  E->setLHS(Record.ReadSubExpr());
  E->setRHS(Record.ReadSubExpr());
  E->setOpcode((BinaryOperator::Opcode)Record[Idx++]);
  E->setOperatorLoc(ReadSourceLocation());
  E->setFPContractable((bool)Record[Idx++]);
}

void ASTStmtReader::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  VisitBinaryOperator(E);
  E->setComputationLHSType(Record.readType(Idx));
  E->setComputationResultType(Record.readType(Idx));
}

void ASTStmtReader::VisitConditionalOperator(ConditionalOperator *E) {
  VisitExpr(E);
  E->SubExprs[ConditionalOperator::COND] = Record.ReadSubExpr();
  E->SubExprs[ConditionalOperator::LHS] = Record.ReadSubExpr();
  E->SubExprs[ConditionalOperator::RHS] = Record.ReadSubExpr();
  E->QuestionLoc = ReadSourceLocation();
  E->ColonLoc = ReadSourceLocation();
}

void
ASTStmtReader::VisitBinaryConditionalOperator(BinaryConditionalOperator *E) {
  VisitExpr(E);
  E->OpaqueValue = cast<OpaqueValueExpr>(Record.ReadSubExpr());
  E->SubExprs[BinaryConditionalOperator::COMMON] = Record.ReadSubExpr();
  E->SubExprs[BinaryConditionalOperator::COND] = Record.ReadSubExpr();
  E->SubExprs[BinaryConditionalOperator::LHS] = Record.ReadSubExpr();
  E->SubExprs[BinaryConditionalOperator::RHS] = Record.ReadSubExpr();
  E->QuestionLoc = ReadSourceLocation();
  E->ColonLoc = ReadSourceLocation();
}

void ASTStmtReader::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  VisitCastExpr(E);
}

void ASTStmtReader::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  VisitCastExpr(E);
  E->setTypeInfoAsWritten(GetTypeSourceInfo());
}

void ASTStmtReader::VisitCStyleCastExpr(CStyleCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->setLParenLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(ReadSourceLocation());
  E->setTypeSourceInfo(GetTypeSourceInfo());
  E->setInitializer(Record.ReadSubExpr());
  E->setFileScope(Record[Idx++]);
}

void ASTStmtReader::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  VisitExpr(E);
  E->setBase(Record.ReadSubExpr());
  E->setAccessor(Record.GetIdentifierInfo(Idx));
  E->setAccessorLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitInitListExpr(InitListExpr *E) {
  VisitExpr(E);
  if (InitListExpr *SyntForm = cast_or_null<InitListExpr>(Record.ReadSubStmt()))
    E->setSyntacticForm(SyntForm);
  E->setLBraceLoc(ReadSourceLocation());
  E->setRBraceLoc(ReadSourceLocation());
  bool isArrayFiller = Record[Idx++];
  Expr *filler = nullptr;
  if (isArrayFiller) {
    filler = Record.ReadSubExpr();
    E->ArrayFillerOrUnionFieldInit = filler;
  } else
    E->ArrayFillerOrUnionFieldInit = ReadDeclAs<FieldDecl>();
  E->sawArrayRangeDesignator(Record[Idx++]);
  unsigned NumInits = Record[Idx++];
  E->reserveInits(Record.getContext(), NumInits);
  if (isArrayFiller) {
    for (unsigned I = 0; I != NumInits; ++I) {
      Expr *init = Record.ReadSubExpr();
      E->updateInit(Record.getContext(), I, init ? init : filler);
    }
  } else {
    for (unsigned I = 0; I != NumInits; ++I)
      E->updateInit(Record.getContext(), I, Record.ReadSubExpr());
  }
}

void ASTStmtReader::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  typedef DesignatedInitExpr::Designator Designator;

  VisitExpr(E);
  unsigned NumSubExprs = Record[Idx++];
  assert(NumSubExprs == E->getNumSubExprs() && "Wrong number of subexprs");
  for (unsigned I = 0; I != NumSubExprs; ++I)
    E->setSubExpr(I, Record.ReadSubExpr());
  E->setEqualOrColonLoc(ReadSourceLocation());
  E->setGNUSyntax(Record[Idx++]);

  SmallVector<Designator, 4> Designators;
  while (Idx < Record.size()) {
    switch ((DesignatorTypes)Record[Idx++]) {
    case DESIG_FIELD_DECL: {
      FieldDecl *Field = ReadDeclAs<FieldDecl>();
      SourceLocation DotLoc = ReadSourceLocation();
      SourceLocation FieldLoc = ReadSourceLocation();
      Designators.push_back(Designator(Field->getIdentifier(), DotLoc,
                                       FieldLoc));
      Designators.back().setField(Field);
      break;
    }

    case DESIG_FIELD_NAME: {
      const IdentifierInfo *Name = Record.GetIdentifierInfo(Idx);
      SourceLocation DotLoc = ReadSourceLocation();
      SourceLocation FieldLoc = ReadSourceLocation();
      Designators.push_back(Designator(Name, DotLoc, FieldLoc));
      break;
    }

    case DESIG_ARRAY: {
      unsigned Index = Record[Idx++];
      SourceLocation LBracketLoc = ReadSourceLocation();
      SourceLocation RBracketLoc = ReadSourceLocation();
      Designators.push_back(Designator(Index, LBracketLoc, RBracketLoc));
      break;
    }

    case DESIG_ARRAY_RANGE: {
      unsigned Index = Record[Idx++];
      SourceLocation LBracketLoc = ReadSourceLocation();
      SourceLocation EllipsisLoc = ReadSourceLocation();
      SourceLocation RBracketLoc = ReadSourceLocation();
      Designators.push_back(Designator(Index, LBracketLoc, EllipsisLoc,
                                       RBracketLoc));
      break;
    }
    }
  }
  E->setDesignators(Record.getContext(),
                    Designators.data(), Designators.size());
}

void ASTStmtReader::VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *E) {
  VisitExpr(E);
  E->setBase(Record.ReadSubExpr());
  E->setUpdater(Record.ReadSubExpr());
}

void ASTStmtReader::VisitNoInitExpr(NoInitExpr *E) {
  VisitExpr(E);
}

void ASTStmtReader::VisitArrayInitLoopExpr(ArrayInitLoopExpr *E) {
  VisitExpr(E);
  E->SubExprs[0] = Record.ReadSubExpr();
  E->SubExprs[1] = Record.ReadSubExpr();
}

void ASTStmtReader::VisitArrayInitIndexExpr(ArrayInitIndexExpr *E) {
  VisitExpr(E);
}

void ASTStmtReader::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  VisitExpr(E);
}

void ASTStmtReader::VisitVAArgExpr(VAArgExpr *E) {
  VisitExpr(E);
  E->setSubExpr(Record.ReadSubExpr());
  E->setWrittenTypeInfo(GetTypeSourceInfo());
  E->setBuiltinLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
  E->setIsMicrosoftABI(Record[Idx++]);
}

void ASTStmtReader::VisitAddrLabelExpr(AddrLabelExpr *E) {
  VisitExpr(E);
  E->setAmpAmpLoc(ReadSourceLocation());
  E->setLabelLoc(ReadSourceLocation());
  E->setLabel(ReadDeclAs<LabelDecl>());
}

void ASTStmtReader::VisitStmtExpr(StmtExpr *E) {
  VisitExpr(E);
  E->setLParenLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
  E->setSubStmt(cast_or_null<CompoundStmt>(Record.ReadSubStmt()));
}

void ASTStmtReader::VisitChooseExpr(ChooseExpr *E) {
  VisitExpr(E);
  E->setCond(Record.ReadSubExpr());
  E->setLHS(Record.ReadSubExpr());
  E->setRHS(Record.ReadSubExpr());
  E->setBuiltinLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
  E->setIsConditionTrue(Record[Idx++]);
}

void ASTStmtReader::VisitGNUNullExpr(GNUNullExpr *E) {
  VisitExpr(E);
  E->setTokenLocation(ReadSourceLocation());
}

void ASTStmtReader::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  VisitExpr(E);
  SmallVector<Expr *, 16> Exprs;
  unsigned NumExprs = Record[Idx++];
  while (NumExprs--)
    Exprs.push_back(Record.ReadSubExpr());
  E->setExprs(Record.getContext(), Exprs);
  E->setBuiltinLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitConvertVectorExpr(ConvertVectorExpr *E) {
  VisitExpr(E);
  E->BuiltinLoc = ReadSourceLocation();
  E->RParenLoc = ReadSourceLocation();
  E->TInfo = GetTypeSourceInfo();
  E->SrcExpr = Record.ReadSubExpr();
}

void ASTStmtReader::VisitBlockExpr(BlockExpr *E) {
  VisitExpr(E);
  E->setBlockDecl(ReadDeclAs<BlockDecl>());
}

void ASTStmtReader::VisitGenericSelectionExpr(GenericSelectionExpr *E) {
  VisitExpr(E);
  E->NumAssocs = Record[Idx++];
  E->AssocTypes = new (Record.getContext()) TypeSourceInfo*[E->NumAssocs];
  E->SubExprs =
   new(Record.getContext()) Stmt*[GenericSelectionExpr::END_EXPR+E->NumAssocs];

  E->SubExprs[GenericSelectionExpr::CONTROLLING] = Record.ReadSubExpr();
  for (unsigned I = 0, N = E->getNumAssocs(); I != N; ++I) {
    E->AssocTypes[I] = GetTypeSourceInfo();
    E->SubExprs[GenericSelectionExpr::END_EXPR+I] = Record.ReadSubExpr();
  }
  E->ResultIndex = Record[Idx++];

  E->GenericLoc = ReadSourceLocation();
  E->DefaultLoc = ReadSourceLocation();
  E->RParenLoc = ReadSourceLocation();
}

void ASTStmtReader::VisitPseudoObjectExpr(PseudoObjectExpr *E) {
  VisitExpr(E);
  unsigned numSemanticExprs = Record[Idx++];
  assert(numSemanticExprs + 1 == E->PseudoObjectExprBits.NumSubExprs);
  E->PseudoObjectExprBits.ResultIndex = Record[Idx++];

  // Read the syntactic expression.
  E->getSubExprsBuffer()[0] = Record.ReadSubExpr();

  // Read all the semantic expressions.
  for (unsigned i = 0; i != numSemanticExprs; ++i) {
    Expr *subExpr = Record.ReadSubExpr();
    E->getSubExprsBuffer()[i+1] = subExpr;
  }
}

void ASTStmtReader::VisitAtomicExpr(AtomicExpr *E) {
  VisitExpr(E);
  E->Op = AtomicExpr::AtomicOp(Record[Idx++]);
  E->NumSubExprs = AtomicExpr::getNumSubExprs(E->Op);
  for (unsigned I = 0; I != E->NumSubExprs; ++I)
    E->SubExprs[I] = Record.ReadSubExpr();
  E->BuiltinLoc = ReadSourceLocation();
  E->RParenLoc = ReadSourceLocation();
}

//===----------------------------------------------------------------------===//
// Objective-C Expressions and Statements

void ASTStmtReader::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  VisitExpr(E);
  E->setString(cast<StringLiteral>(Record.ReadSubStmt()));
  E->setAtLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCBoxedExpr(ObjCBoxedExpr *E) {
  VisitExpr(E);
  // could be one of several IntegerLiteral, FloatLiteral, etc.
  E->SubExpr = Record.ReadSubStmt();
  E->BoxingMethod = ReadDeclAs<ObjCMethodDecl>();
  E->Range = ReadSourceRange();
}

void ASTStmtReader::VisitObjCArrayLiteral(ObjCArrayLiteral *E) {
  VisitExpr(E);
  unsigned NumElements = Record[Idx++];
  assert(NumElements == E->getNumElements() && "Wrong number of elements");
  Expr **Elements = E->getElements();
  for (unsigned I = 0, N = NumElements; I != N; ++I)
    Elements[I] = Record.ReadSubExpr();
  E->ArrayWithObjectsMethod = ReadDeclAs<ObjCMethodDecl>();
  E->Range = ReadSourceRange();
}

void ASTStmtReader::VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *E) {
  VisitExpr(E);
  unsigned NumElements = Record[Idx++];
  assert(NumElements == E->getNumElements() && "Wrong number of elements");
  bool HasPackExpansions = Record[Idx++];
  assert(HasPackExpansions == E->HasPackExpansions &&"Pack expansion mismatch");
  ObjCDictionaryLiteral::KeyValuePair *KeyValues =
      E->getTrailingObjects<ObjCDictionaryLiteral::KeyValuePair>();
  ObjCDictionaryLiteral::ExpansionData *Expansions =
      E->getTrailingObjects<ObjCDictionaryLiteral::ExpansionData>();
  for (unsigned I = 0; I != NumElements; ++I) {
    KeyValues[I].Key = Record.ReadSubExpr();
    KeyValues[I].Value = Record.ReadSubExpr();
    if (HasPackExpansions) {
      Expansions[I].EllipsisLoc = ReadSourceLocation();
      Expansions[I].NumExpansionsPlusOne = Record[Idx++];
    }
  }
  E->DictWithObjectsMethod = ReadDeclAs<ObjCMethodDecl>();
  E->Range = ReadSourceRange();
}

void ASTStmtReader::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  VisitExpr(E);
  E->setEncodedTypeSourceInfo(GetTypeSourceInfo());
  E->setAtLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
  VisitExpr(E);
  E->setSelector(Record.ReadSelector(Idx));
  E->setAtLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
  VisitExpr(E);
  E->setProtocol(ReadDeclAs<ObjCProtocolDecl>());
  E->setAtLoc(ReadSourceLocation());
  E->ProtoLoc = ReadSourceLocation();
  E->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  VisitExpr(E);
  E->setDecl(ReadDeclAs<ObjCIvarDecl>());
  E->setLocation(ReadSourceLocation());
  E->setOpLoc(ReadSourceLocation());
  E->setBase(Record.ReadSubExpr());
  E->setIsArrow(Record[Idx++]);
  E->setIsFreeIvar(Record[Idx++]);
}

void ASTStmtReader::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  VisitExpr(E);
  unsigned MethodRefFlags = Record[Idx++];
  bool Implicit = Record[Idx++] != 0;
  if (Implicit) {
    ObjCMethodDecl *Getter = ReadDeclAs<ObjCMethodDecl>();
    ObjCMethodDecl *Setter = ReadDeclAs<ObjCMethodDecl>();
    E->setImplicitProperty(Getter, Setter, MethodRefFlags);
  } else {
    E->setExplicitProperty(ReadDeclAs<ObjCPropertyDecl>(), MethodRefFlags);
  }
  E->setLocation(ReadSourceLocation());
  E->setReceiverLocation(ReadSourceLocation());
  switch (Record[Idx++]) {
  case 0:
    E->setBase(Record.ReadSubExpr());
    break;
  case 1:
    E->setSuperReceiver(Record.readType(Idx));
    break;
  case 2:
    E->setClassReceiver(ReadDeclAs<ObjCInterfaceDecl>());
    break;
  }
}

void ASTStmtReader::VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *E) {
  VisitExpr(E);
  E->setRBracket(ReadSourceLocation());
  E->setBaseExpr(Record.ReadSubExpr());
  E->setKeyExpr(Record.ReadSubExpr());
  E->GetAtIndexMethodDecl = ReadDeclAs<ObjCMethodDecl>();
  E->SetAtIndexMethodDecl = ReadDeclAs<ObjCMethodDecl>();
}

void ASTStmtReader::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  VisitExpr(E);
  assert(Record[Idx] == E->getNumArgs());
  ++Idx;
  unsigned NumStoredSelLocs = Record[Idx++];
  E->SelLocsKind = Record[Idx++];
  E->setDelegateInitCall(Record[Idx++]);
  E->IsImplicit = Record[Idx++];
  ObjCMessageExpr::ReceiverKind Kind
    = static_cast<ObjCMessageExpr::ReceiverKind>(Record[Idx++]);
  switch (Kind) {
  case ObjCMessageExpr::Instance:
    E->setInstanceReceiver(Record.ReadSubExpr());
    break;

  case ObjCMessageExpr::Class:
    E->setClassReceiver(GetTypeSourceInfo());
    break;

  case ObjCMessageExpr::SuperClass:
  case ObjCMessageExpr::SuperInstance: {
    QualType T = Record.readType(Idx);
    SourceLocation SuperLoc = ReadSourceLocation();
    E->setSuper(SuperLoc, T, Kind == ObjCMessageExpr::SuperInstance);
    break;
  }
  }

  assert(Kind == E->getReceiverKind());

  if (Record[Idx++])
    E->setMethodDecl(ReadDeclAs<ObjCMethodDecl>());
  else
    E->setSelector(Record.ReadSelector(Idx));

  E->LBracLoc = ReadSourceLocation();
  E->RBracLoc = ReadSourceLocation();

  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, Record.ReadSubExpr());

  SourceLocation *Locs = E->getStoredSelLocs();
  for (unsigned I = 0; I != NumStoredSelLocs; ++I)
    Locs[I] = ReadSourceLocation();
}

void ASTStmtReader::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  VisitStmt(S);
  S->setElement(Record.ReadSubStmt());
  S->setCollection(Record.ReadSubExpr());
  S->setBody(Record.ReadSubStmt());
  S->setForLoc(ReadSourceLocation());
  S->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  VisitStmt(S);
  S->setCatchBody(Record.ReadSubStmt());
  S->setCatchParamDecl(ReadDeclAs<VarDecl>());
  S->setAtCatchLoc(ReadSourceLocation());
  S->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  VisitStmt(S);
  S->setFinallyBody(Record.ReadSubStmt());
  S->setAtFinallyLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *S) {
  VisitStmt(S);
  S->setSubStmt(Record.ReadSubStmt());
  S->setAtLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  VisitStmt(S);
  assert(Record[Idx] == S->getNumCatchStmts());
  ++Idx;
  bool HasFinally = Record[Idx++];
  S->setTryBody(Record.ReadSubStmt());
  for (unsigned I = 0, N = S->getNumCatchStmts(); I != N; ++I)
    S->setCatchStmt(I, cast_or_null<ObjCAtCatchStmt>(Record.ReadSubStmt()));

  if (HasFinally)
    S->setFinallyStmt(Record.ReadSubStmt());
  S->setAtTryLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S) {
  VisitStmt(S);
  S->setSynchExpr(Record.ReadSubStmt());
  S->setSynchBody(Record.ReadSubStmt());
  S->setAtSynchronizedLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  VisitStmt(S);
  S->setThrowExpr(Record.ReadSubStmt());
  S->setThrowLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *E) {
  VisitExpr(E);
  E->setValue(Record[Idx++]);
  E->setLocation(ReadSourceLocation());
}

void ASTStmtReader::VisitObjCAvailabilityCheckExpr(ObjCAvailabilityCheckExpr *E) {
  VisitExpr(E);
  SourceRange R = Record.ReadSourceRange(Idx);
  E->AtLoc = R.getBegin();
  E->RParen = R.getEnd();
  E->VersionToCheck = Record.ReadVersionTuple(Idx);
}

//===----------------------------------------------------------------------===//
// C++ Expressions and Statements
//===----------------------------------------------------------------------===//

void ASTStmtReader::VisitCXXCatchStmt(CXXCatchStmt *S) {
  VisitStmt(S);
  S->CatchLoc = ReadSourceLocation();
  S->ExceptionDecl = ReadDeclAs<VarDecl>();
  S->HandlerBlock = Record.ReadSubStmt();
}

void ASTStmtReader::VisitCXXTryStmt(CXXTryStmt *S) {
  VisitStmt(S);
  assert(Record[Idx] == S->getNumHandlers() && "NumStmtFields is wrong ?");
  ++Idx;
  S->TryLoc = ReadSourceLocation();
  S->getStmts()[0] = Record.ReadSubStmt();
  for (unsigned i = 0, e = S->getNumHandlers(); i != e; ++i)
    S->getStmts()[i + 1] = Record.ReadSubStmt();
}

void ASTStmtReader::VisitCXXForRangeStmt(CXXForRangeStmt *S) {
  VisitStmt(S);
  S->ForLoc = ReadSourceLocation();
  S->CoawaitLoc = ReadSourceLocation();
  S->ColonLoc = ReadSourceLocation();
  S->RParenLoc = ReadSourceLocation();
  S->setRangeStmt(Record.ReadSubStmt());
  S->setBeginStmt(Record.ReadSubStmt());
  S->setEndStmt(Record.ReadSubStmt());
  S->setCond(Record.ReadSubExpr());
  S->setInc(Record.ReadSubExpr());
  S->setLoopVarStmt(Record.ReadSubStmt());
  S->setBody(Record.ReadSubStmt());
}

void ASTStmtReader::VisitMSDependentExistsStmt(MSDependentExistsStmt *S) {
  VisitStmt(S);
  S->KeywordLoc = ReadSourceLocation();
  S->IsIfExists = Record[Idx++];
  S->QualifierLoc = Record.ReadNestedNameSpecifierLoc(Idx);
  ReadDeclarationNameInfo(S->NameInfo);
  S->SubStmt = Record.ReadSubStmt();
}

void ASTStmtReader::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  VisitCallExpr(E);
  E->Operator = (OverloadedOperatorKind)Record[Idx++];
  E->Range = Record.ReadSourceRange(Idx);
  E->setFPContractable((bool)Record[Idx++]);
}

void ASTStmtReader::VisitCXXConstructExpr(CXXConstructExpr *E) {
  VisitExpr(E);
  E->NumArgs = Record[Idx++];
  if (E->NumArgs)
    E->Args = new (Record.getContext()) Stmt*[E->NumArgs];
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    E->setArg(I, Record.ReadSubExpr());
  E->setConstructor(ReadDeclAs<CXXConstructorDecl>());
  E->setLocation(ReadSourceLocation());
  E->setElidable(Record[Idx++]);
  E->setHadMultipleCandidates(Record[Idx++]);
  E->setListInitialization(Record[Idx++]);
  E->setStdInitListInitialization(Record[Idx++]);
  E->setRequiresZeroInitialization(Record[Idx++]);
  E->setConstructionKind((CXXConstructExpr::ConstructionKind)Record[Idx++]);
  E->ParenOrBraceRange = ReadSourceRange();
}

void ASTStmtReader::VisitCXXInheritedCtorInitExpr(CXXInheritedCtorInitExpr *E) {
  VisitExpr(E);
  E->Constructor = ReadDeclAs<CXXConstructorDecl>();
  E->Loc = ReadSourceLocation();
  E->ConstructsVirtualBase = Record[Idx++];
  E->InheritedFromVirtualBase = Record[Idx++];
}

void ASTStmtReader::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E) {
  VisitCXXConstructExpr(E);
  E->Type = GetTypeSourceInfo();
}

void ASTStmtReader::VisitLambdaExpr(LambdaExpr *E) {
  VisitExpr(E);
  unsigned NumCaptures = Record[Idx++];
  assert(NumCaptures == E->NumCaptures);(void)NumCaptures;
  E->IntroducerRange = ReadSourceRange();
  E->CaptureDefault = static_cast<LambdaCaptureDefault>(Record[Idx++]);
  E->CaptureDefaultLoc = ReadSourceLocation();
  E->ExplicitParams = Record[Idx++];
  E->ExplicitResultType = Record[Idx++];
  E->ClosingBrace = ReadSourceLocation();

  // Read capture initializers.
  for (LambdaExpr::capture_init_iterator C = E->capture_init_begin(),
                                      CEnd = E->capture_init_end();
       C != CEnd; ++C)
    *C = Record.ReadSubExpr();
}

void
ASTStmtReader::VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *E) {
  VisitExpr(E);
  E->SubExpr = Record.ReadSubExpr();
}

void ASTStmtReader::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  VisitExplicitCastExpr(E);
  SourceRange R = ReadSourceRange();
  E->Loc = R.getBegin();
  E->RParenLoc = R.getEnd();
  R = ReadSourceRange();
  E->AngleBrackets = R;
}

void ASTStmtReader::VisitCXXStaticCastExpr(CXXStaticCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

void ASTStmtReader::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

void ASTStmtReader::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

void ASTStmtReader::VisitCXXConstCastExpr(CXXConstCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

void ASTStmtReader::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *E) {
  VisitExplicitCastExpr(E);
  E->setLParenLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitUserDefinedLiteral(UserDefinedLiteral *E) {
  VisitCallExpr(E);
  E->UDSuffixLoc = ReadSourceLocation();
}

void ASTStmtReader::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  VisitExpr(E);
  E->setValue(Record[Idx++]);
  E->setLocation(ReadSourceLocation());
}

void ASTStmtReader::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  VisitExpr(E);
  E->setLocation(ReadSourceLocation());
}

void ASTStmtReader::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  VisitExpr(E);
  E->setSourceRange(ReadSourceRange());
  if (E->isTypeOperand()) { // typeid(int)
    E->setTypeOperandSourceInfo(
        GetTypeSourceInfo());
    return;
  }

  // typeid(42+2)
  E->setExprOperand(Record.ReadSubExpr());
}

void ASTStmtReader::VisitCXXThisExpr(CXXThisExpr *E) {
  VisitExpr(E);
  E->setLocation(ReadSourceLocation());
  E->setImplicit(Record[Idx++]);
}

void ASTStmtReader::VisitCXXThrowExpr(CXXThrowExpr *E) {
  VisitExpr(E);
  E->ThrowLoc = ReadSourceLocation();
  E->Op = Record.ReadSubExpr();
  E->IsThrownVariableInScope = Record[Idx++];
}

void ASTStmtReader::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  VisitExpr(E);
  E->Param = ReadDeclAs<ParmVarDecl>();
  E->Loc = ReadSourceLocation();
}

void ASTStmtReader::VisitCXXDefaultInitExpr(CXXDefaultInitExpr *E) {
  VisitExpr(E);
  E->Field = ReadDeclAs<FieldDecl>();
  E->Loc = ReadSourceLocation();
}

void ASTStmtReader::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  VisitExpr(E);
  E->setTemporary(Record.ReadCXXTemporary(Idx));
  E->setSubExpr(Record.ReadSubExpr());
}

void ASTStmtReader::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  VisitExpr(E);
  E->TypeInfo = GetTypeSourceInfo();
  E->RParenLoc = ReadSourceLocation();
}

void ASTStmtReader::VisitCXXNewExpr(CXXNewExpr *E) {
  VisitExpr(E);
  E->GlobalNew = Record[Idx++];
  bool isArray = Record[Idx++];
  E->PassAlignment = Record[Idx++];
  E->UsualArrayDeleteWantsSize = Record[Idx++];
  unsigned NumPlacementArgs = Record[Idx++];
  E->StoredInitializationStyle = Record[Idx++];
  E->setOperatorNew(ReadDeclAs<FunctionDecl>());
  E->setOperatorDelete(ReadDeclAs<FunctionDecl>());
  E->AllocatedTypeInfo = GetTypeSourceInfo();
  E->TypeIdParens = ReadSourceRange();
  E->Range = ReadSourceRange();
  E->DirectInitRange = ReadSourceRange();

  E->AllocateArgsArray(Record.getContext(), isArray, NumPlacementArgs,
                       E->StoredInitializationStyle != 0);

  // Install all the subexpressions.
  for (CXXNewExpr::raw_arg_iterator I = E->raw_arg_begin(),e = E->raw_arg_end();
       I != e; ++I)
    *I = Record.ReadSubStmt();
}

void ASTStmtReader::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  VisitExpr(E);
  E->GlobalDelete = Record[Idx++];
  E->ArrayForm = Record[Idx++];
  E->ArrayFormAsWritten = Record[Idx++];
  E->UsualArrayDeleteWantsSize = Record[Idx++];
  E->OperatorDelete = ReadDeclAs<FunctionDecl>();
  E->Argument = Record.ReadSubExpr();
  E->Loc = ReadSourceLocation();
}

void ASTStmtReader::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  VisitExpr(E);

  E->Base = Record.ReadSubExpr();
  E->IsArrow = Record[Idx++];
  E->OperatorLoc = ReadSourceLocation();
  E->QualifierLoc = Record.ReadNestedNameSpecifierLoc(Idx);
  E->ScopeType = GetTypeSourceInfo();
  E->ColonColonLoc = ReadSourceLocation();
  E->TildeLoc = ReadSourceLocation();

  IdentifierInfo *II = Record.GetIdentifierInfo(Idx);
  if (II)
    E->setDestroyedType(II, ReadSourceLocation());
  else
    E->setDestroyedType(GetTypeSourceInfo());
}

void ASTStmtReader::VisitExprWithCleanups(ExprWithCleanups *E) {
  VisitExpr(E);

  unsigned NumObjects = Record[Idx++];
  assert(NumObjects == E->getNumObjects());
  for (unsigned i = 0; i != NumObjects; ++i)
    E->getTrailingObjects<BlockDecl *>()[i] =
        ReadDeclAs<BlockDecl>();

  E->ExprWithCleanupsBits.CleanupsHaveSideEffects = Record[Idx++];
  E->SubExpr = Record.ReadSubExpr();
}

void
ASTStmtReader::VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E){
  VisitExpr(E);

  if (Record[Idx++]) // HasTemplateKWAndArgsInfo
    ReadTemplateKWAndArgsInfo(
        *E->getTrailingObjects<ASTTemplateKWAndArgsInfo>(),
        E->getTrailingObjects<TemplateArgumentLoc>(),
        /*NumTemplateArgs=*/Record[Idx++]);

  E->Base = Record.ReadSubExpr();
  E->BaseType = Record.readType(Idx);
  E->IsArrow = Record[Idx++];
  E->OperatorLoc = ReadSourceLocation();
  E->QualifierLoc = Record.ReadNestedNameSpecifierLoc(Idx);
  E->FirstQualifierFoundInScope = ReadDeclAs<NamedDecl>();
  ReadDeclarationNameInfo(E->MemberNameInfo);
}

void
ASTStmtReader::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) {
  VisitExpr(E);

  if (Record[Idx++]) // HasTemplateKWAndArgsInfo
    ReadTemplateKWAndArgsInfo(
        *E->getTrailingObjects<ASTTemplateKWAndArgsInfo>(),
        E->getTrailingObjects<TemplateArgumentLoc>(),
        /*NumTemplateArgs=*/Record[Idx++]);

  E->QualifierLoc = Record.ReadNestedNameSpecifierLoc(Idx);
  ReadDeclarationNameInfo(E->NameInfo);
}

void
ASTStmtReader::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E) {
  VisitExpr(E);
  assert(Record[Idx] == E->arg_size() && "Read wrong record during creation ?");
  ++Idx; // NumArgs;
  for (unsigned I = 0, N = E->arg_size(); I != N; ++I)
    E->setArg(I, Record.ReadSubExpr());
  E->Type = GetTypeSourceInfo();
  E->setLParenLoc(ReadSourceLocation());
  E->setRParenLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitOverloadExpr(OverloadExpr *E) {
  VisitExpr(E);

  if (Record[Idx++]) // HasTemplateKWAndArgsInfo
    ReadTemplateKWAndArgsInfo(*E->getTrailingASTTemplateKWAndArgsInfo(),
                              E->getTrailingTemplateArgumentLoc(),
                              /*NumTemplateArgs=*/Record[Idx++]);

  unsigned NumDecls = Record[Idx++];
  UnresolvedSet<8> Decls;
  for (unsigned i = 0; i != NumDecls; ++i) {
    NamedDecl *D = ReadDeclAs<NamedDecl>();
    AccessSpecifier AS = (AccessSpecifier)Record[Idx++];
    Decls.addDecl(D, AS);
  }
  E->initializeResults(Record.getContext(), Decls.begin(), Decls.end());

  ReadDeclarationNameInfo(E->NameInfo);
  E->QualifierLoc = Record.ReadNestedNameSpecifierLoc(Idx);
}

void ASTStmtReader::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E) {
  VisitOverloadExpr(E);
  E->IsArrow = Record[Idx++];
  E->HasUnresolvedUsing = Record[Idx++];
  E->Base = Record.ReadSubExpr();
  E->BaseType = Record.readType(Idx);
  E->OperatorLoc = ReadSourceLocation();
}

void ASTStmtReader::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E) {
  VisitOverloadExpr(E);
  E->RequiresADL = Record[Idx++];
  E->Overloaded = Record[Idx++];
  E->NamingClass = ReadDeclAs<CXXRecordDecl>();
}

void ASTStmtReader::VisitTypeTraitExpr(TypeTraitExpr *E) {
  VisitExpr(E);
  E->TypeTraitExprBits.NumArgs = Record[Idx++];
  E->TypeTraitExprBits.Kind = Record[Idx++];
  E->TypeTraitExprBits.Value = Record[Idx++];
  SourceRange Range = ReadSourceRange();
  E->Loc = Range.getBegin();
  E->RParenLoc = Range.getEnd();

  TypeSourceInfo **Args = E->getTrailingObjects<TypeSourceInfo *>();
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I)
    Args[I] = GetTypeSourceInfo();
}

void ASTStmtReader::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  VisitExpr(E);
  E->ATT = (ArrayTypeTrait)Record[Idx++];
  E->Value = (unsigned int)Record[Idx++];
  SourceRange Range = ReadSourceRange();
  E->Loc = Range.getBegin();
  E->RParen = Range.getEnd();
  E->QueriedType = GetTypeSourceInfo();
  E->Dimension = Record.ReadSubExpr();
}

void ASTStmtReader::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  VisitExpr(E);
  E->ET = (ExpressionTrait)Record[Idx++];
  E->Value = (bool)Record[Idx++];
  SourceRange Range = ReadSourceRange();
  E->QueriedExpression = Record.ReadSubExpr();
  E->Loc = Range.getBegin();
  E->RParen = Range.getEnd();
}

void ASTStmtReader::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  VisitExpr(E);
  E->Value = (bool)Record[Idx++];
  E->Range = ReadSourceRange();
  E->Operand = Record.ReadSubExpr();
}

void ASTStmtReader::VisitPackExpansionExpr(PackExpansionExpr *E) {
  VisitExpr(E);
  E->EllipsisLoc = ReadSourceLocation();
  E->NumExpansions = Record[Idx++];
  E->Pattern = Record.ReadSubExpr();
}

void ASTStmtReader::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  VisitExpr(E);
  unsigned NumPartialArgs = Record[Idx++];
  E->OperatorLoc = ReadSourceLocation();
  E->PackLoc = ReadSourceLocation();
  E->RParenLoc = ReadSourceLocation();
  E->Pack = Record.ReadDeclAs<NamedDecl>(Idx);
  if (E->isPartiallySubstituted()) {
    assert(E->Length == NumPartialArgs);
    for (auto *I = E->getTrailingObjects<TemplateArgument>(),
              *E = I + NumPartialArgs;
         I != E; ++I)
      new (I) TemplateArgument(Record.ReadTemplateArgument(Idx));
  } else if (!E->isValueDependent()) {
    E->Length = Record[Idx++];
  }
}

void ASTStmtReader::VisitSubstNonTypeTemplateParmExpr(
                                              SubstNonTypeTemplateParmExpr *E) {
  VisitExpr(E);
  E->Param = ReadDeclAs<NonTypeTemplateParmDecl>();
  E->NameLoc = ReadSourceLocation();
  E->Replacement = Record.ReadSubExpr();
}

void ASTStmtReader::VisitSubstNonTypeTemplateParmPackExpr(
                                          SubstNonTypeTemplateParmPackExpr *E) {
  VisitExpr(E);
  E->Param = ReadDeclAs<NonTypeTemplateParmDecl>();
  TemplateArgument ArgPack = Record.ReadTemplateArgument(Idx);
  if (ArgPack.getKind() != TemplateArgument::Pack)
    return;

  E->Arguments = ArgPack.pack_begin();
  E->NumArguments = ArgPack.pack_size();
  E->NameLoc = ReadSourceLocation();
}

void ASTStmtReader::VisitFunctionParmPackExpr(FunctionParmPackExpr *E) {
  VisitExpr(E);
  E->NumParameters = Record[Idx++];
  E->ParamPack = ReadDeclAs<ParmVarDecl>();
  E->NameLoc = ReadSourceLocation();
  ParmVarDecl **Parms = E->getTrailingObjects<ParmVarDecl *>();
  for (unsigned i = 0, n = E->NumParameters; i != n; ++i)
    Parms[i] = ReadDeclAs<ParmVarDecl>();
}

void ASTStmtReader::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E) {
  VisitExpr(E);
  E->State = Record.ReadSubExpr();
  auto VD = ReadDeclAs<ValueDecl>();
  unsigned ManglingNumber = Record[Idx++];
  E->setExtendingDecl(VD, ManglingNumber);
}

void ASTStmtReader::VisitCXXFoldExpr(CXXFoldExpr *E) {
  VisitExpr(E);
  E->LParenLoc = ReadSourceLocation();
  E->EllipsisLoc = ReadSourceLocation();
  E->RParenLoc = ReadSourceLocation();
  E->SubExprs[0] = Record.ReadSubExpr();
  E->SubExprs[1] = Record.ReadSubExpr();
  E->Opcode = (BinaryOperatorKind)Record[Idx++];
}

void ASTStmtReader::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  VisitExpr(E);
  E->SourceExpr = Record.ReadSubExpr();
  E->Loc = ReadSourceLocation();
}

void ASTStmtReader::VisitTypoExpr(TypoExpr *E) {
  llvm_unreachable("Cannot read TypoExpr nodes");
}

//===----------------------------------------------------------------------===//
// Microsoft Expressions and Statements
//===----------------------------------------------------------------------===//
void ASTStmtReader::VisitMSPropertyRefExpr(MSPropertyRefExpr *E) {
  VisitExpr(E);
  E->IsArrow = (Record[Idx++] != 0);
  E->BaseExpr = Record.ReadSubExpr();
  E->QualifierLoc = Record.ReadNestedNameSpecifierLoc(Idx);
  E->MemberLoc = ReadSourceLocation();
  E->TheDecl = ReadDeclAs<MSPropertyDecl>();
}

void ASTStmtReader::VisitMSPropertySubscriptExpr(MSPropertySubscriptExpr *E) {
  VisitExpr(E);
  E->setBase(Record.ReadSubExpr());
  E->setIdx(Record.ReadSubExpr());
  E->setRBracketLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitCXXUuidofExpr(CXXUuidofExpr *E) {
  VisitExpr(E);
  E->setSourceRange(ReadSourceRange());
  std::string UuidStr = ReadString();
  E->setUuidStr(StringRef(UuidStr).copy(Record.getContext()));
  if (E->isTypeOperand()) { // __uuidof(ComType)
    E->setTypeOperandSourceInfo(
        GetTypeSourceInfo());
    return;
  }

  // __uuidof(expr)
  E->setExprOperand(Record.ReadSubExpr());
}

void ASTStmtReader::VisitSEHLeaveStmt(SEHLeaveStmt *S) {
  VisitStmt(S);
  S->setLeaveLoc(ReadSourceLocation());
}

void ASTStmtReader::VisitSEHExceptStmt(SEHExceptStmt *S) {
  VisitStmt(S);
  S->Loc = ReadSourceLocation();
  S->Children[SEHExceptStmt::FILTER_EXPR] = Record.ReadSubStmt();
  S->Children[SEHExceptStmt::BLOCK] = Record.ReadSubStmt();
}

void ASTStmtReader::VisitSEHFinallyStmt(SEHFinallyStmt *S) {
  VisitStmt(S);
  S->Loc = ReadSourceLocation();
  S->Block = Record.ReadSubStmt();
}

void ASTStmtReader::VisitSEHTryStmt(SEHTryStmt *S) {
  VisitStmt(S);
  S->IsCXXTry = Record[Idx++];
  S->TryLoc = ReadSourceLocation();
  S->Children[SEHTryStmt::TRY] = Record.ReadSubStmt();
  S->Children[SEHTryStmt::HANDLER] = Record.ReadSubStmt();
}

//===----------------------------------------------------------------------===//
// CUDA Expressions and Statements
//===----------------------------------------------------------------------===//

void ASTStmtReader::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *E) {
  VisitCallExpr(E);
  E->setConfig(cast<CallExpr>(Record.ReadSubExpr()));
}

//===----------------------------------------------------------------------===//
// OpenCL Expressions and Statements.
//===----------------------------------------------------------------------===//
void ASTStmtReader::VisitAsTypeExpr(AsTypeExpr *E) {
  VisitExpr(E);
  E->BuiltinLoc = ReadSourceLocation();
  E->RParenLoc = ReadSourceLocation();
  E->SrcExpr = Record.ReadSubExpr();
}

//===----------------------------------------------------------------------===//
// OpenMP Clauses.
//===----------------------------------------------------------------------===//

namespace clang {
class OMPClauseReader : public OMPClauseVisitor<OMPClauseReader> {
  ASTStmtReader *Reader;
  ASTContext &Context;
  unsigned &Idx;
public:
  OMPClauseReader(ASTStmtReader *R, ASTRecordReader &Record, unsigned &Idx)
      : Reader(R), Context(Record.getContext()), Idx(Idx) {}
#define OPENMP_CLAUSE(Name, Class) void Visit##Class(Class *C);
#include "clang/Basic/OpenMPKinds.def"
  OMPClause *readClause();
  void VisitOMPClauseWithPreInit(OMPClauseWithPreInit *C);
  void VisitOMPClauseWithPostUpdate(OMPClauseWithPostUpdate *C);
};
}

OMPClause *OMPClauseReader::readClause() {
  OMPClause *C;
  switch (Reader->Record[Idx++]) {
  case OMPC_if:
    C = new (Context) OMPIfClause();
    break;
  case OMPC_final:
    C = new (Context) OMPFinalClause();
    break;
  case OMPC_num_threads:
    C = new (Context) OMPNumThreadsClause();
    break;
  case OMPC_safelen:
    C = new (Context) OMPSafelenClause();
    break;
  case OMPC_simdlen:
    C = new (Context) OMPSimdlenClause();
    break;
  case OMPC_collapse:
    C = new (Context) OMPCollapseClause();
    break;
  case OMPC_default:
    C = new (Context) OMPDefaultClause();
    break;
  case OMPC_proc_bind:
    C = new (Context) OMPProcBindClause();
    break;
  case OMPC_schedule:
    C = new (Context) OMPScheduleClause();
    break;
  case OMPC_ordered:
    C = new (Context) OMPOrderedClause();
    break;
  case OMPC_nowait:
    C = new (Context) OMPNowaitClause();
    break;
  case OMPC_untied:
    C = new (Context) OMPUntiedClause();
    break;
  case OMPC_mergeable:
    C = new (Context) OMPMergeableClause();
    break;
  case OMPC_read:
    C = new (Context) OMPReadClause();
    break;
  case OMPC_write:
    C = new (Context) OMPWriteClause();
    break;
  case OMPC_update:
    C = new (Context) OMPUpdateClause();
    break;
  case OMPC_capture:
    C = new (Context) OMPCaptureClause();
    break;
  case OMPC_seq_cst:
    C = new (Context) OMPSeqCstClause();
    break;
  case OMPC_threads:
    C = new (Context) OMPThreadsClause();
    break;
  case OMPC_simd:
    C = new (Context) OMPSIMDClause();
    break;
  case OMPC_nogroup:
    C = new (Context) OMPNogroupClause();
    break;
  case OMPC_private:
    C = OMPPrivateClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_firstprivate:
    C = OMPFirstprivateClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_lastprivate:
    C = OMPLastprivateClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_shared:
    C = OMPSharedClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_reduction:
    C = OMPReductionClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_linear:
    C = OMPLinearClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_aligned:
    C = OMPAlignedClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_copyin:
    C = OMPCopyinClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_copyprivate:
    C = OMPCopyprivateClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_flush:
    C = OMPFlushClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_depend:
    C = OMPDependClause::CreateEmpty(Context, Reader->Record[Idx++]);
    break;
  case OMPC_device:
    C = new (Context) OMPDeviceClause();
    break;
  case OMPC_map: {
    unsigned NumVars = Reader->Record[Idx++];
    unsigned NumDeclarations = Reader->Record[Idx++];
    unsigned NumLists = Reader->Record[Idx++];
    unsigned NumComponents = Reader->Record[Idx++];
    C = OMPMapClause::CreateEmpty(Context, NumVars, NumDeclarations, NumLists,
                                  NumComponents);
    break;
  }
  case OMPC_num_teams:
    C = new (Context) OMPNumTeamsClause();
    break;
  case OMPC_thread_limit:
    C = new (Context) OMPThreadLimitClause();
    break;
  case OMPC_priority:
    C = new (Context) OMPPriorityClause();
    break;
  case OMPC_grainsize:
    C = new (Context) OMPGrainsizeClause();
    break;
  case OMPC_num_tasks:
    C = new (Context) OMPNumTasksClause();
    break;
  case OMPC_hint:
    C = new (Context) OMPHintClause();
    break;
  case OMPC_dist_schedule:
    C = new (Context) OMPDistScheduleClause();
    break;
  case OMPC_defaultmap:
    C = new (Context) OMPDefaultmapClause();
    break;
  case OMPC_to: {
    unsigned NumVars = Reader->Record[Idx++];
    unsigned NumDeclarations = Reader->Record[Idx++];
    unsigned NumLists = Reader->Record[Idx++];
    unsigned NumComponents = Reader->Record[Idx++];
    C = OMPToClause::CreateEmpty(Context, NumVars, NumDeclarations, NumLists,
                                 NumComponents);
    break;
  }
  case OMPC_from: {
    unsigned NumVars = Reader->Record[Idx++];
    unsigned NumDeclarations = Reader->Record[Idx++];
    unsigned NumLists = Reader->Record[Idx++];
    unsigned NumComponents = Reader->Record[Idx++];
    C = OMPFromClause::CreateEmpty(Context, NumVars, NumDeclarations, NumLists,
                                   NumComponents);
    break;
  }
  case OMPC_use_device_ptr: {
    unsigned NumVars = Reader->Record[Idx++];
    unsigned NumDeclarations = Reader->Record[Idx++];
    unsigned NumLists = Reader->Record[Idx++];
    unsigned NumComponents = Reader->Record[Idx++];
    C = OMPUseDevicePtrClause::CreateEmpty(Context, NumVars, NumDeclarations,
                                           NumLists, NumComponents);
    break;
  }
  case OMPC_is_device_ptr: {
    unsigned NumVars = Reader->Record[Idx++];
    unsigned NumDeclarations = Reader->Record[Idx++];
    unsigned NumLists = Reader->Record[Idx++];
    unsigned NumComponents = Reader->Record[Idx++];
    C = OMPIsDevicePtrClause::CreateEmpty(Context, NumVars, NumDeclarations,
                                          NumLists, NumComponents);
    break;
  }
  }
  Visit(C);
  C->setLocStart(Reader->ReadSourceLocation());
  C->setLocEnd(Reader->ReadSourceLocation());

  return C;
}

void OMPClauseReader::VisitOMPClauseWithPreInit(OMPClauseWithPreInit *C) {
  C->setPreInitStmt(Reader->Record.ReadSubStmt());
}

void OMPClauseReader::VisitOMPClauseWithPostUpdate(OMPClauseWithPostUpdate *C) {
  VisitOMPClauseWithPreInit(C);
  C->setPostUpdateExpr(Reader->Record.ReadSubExpr());
}

void OMPClauseReader::VisitOMPIfClause(OMPIfClause *C) {
  C->setNameModifier(static_cast<OpenMPDirectiveKind>(Reader->Record[Idx++]));
  C->setNameModifierLoc(Reader->ReadSourceLocation());
  C->setColonLoc(Reader->ReadSourceLocation());
  C->setCondition(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPFinalClause(OMPFinalClause *C) {
  C->setCondition(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPNumThreadsClause(OMPNumThreadsClause *C) {
  C->setNumThreads(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPSafelenClause(OMPSafelenClause *C) {
  C->setSafelen(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPSimdlenClause(OMPSimdlenClause *C) {
  C->setSimdlen(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPCollapseClause(OMPCollapseClause *C) {
  C->setNumForLoops(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPDefaultClause(OMPDefaultClause *C) {
  C->setDefaultKind(
       static_cast<OpenMPDefaultClauseKind>(Reader->Record[Idx++]));
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setDefaultKindKwLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPProcBindClause(OMPProcBindClause *C) {
  C->setProcBindKind(
       static_cast<OpenMPProcBindClauseKind>(Reader->Record[Idx++]));
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setProcBindKindKwLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPScheduleClause(OMPScheduleClause *C) {
  VisitOMPClauseWithPreInit(C);
  C->setScheduleKind(
       static_cast<OpenMPScheduleClauseKind>(Reader->Record[Idx++]));
  C->setFirstScheduleModifier(
      static_cast<OpenMPScheduleClauseModifier>(Reader->Record[Idx++]));
  C->setSecondScheduleModifier(
      static_cast<OpenMPScheduleClauseModifier>(Reader->Record[Idx++]));
  C->setChunkSize(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setFirstScheduleModifierLoc(Reader->ReadSourceLocation());
  C->setSecondScheduleModifierLoc(Reader->ReadSourceLocation());
  C->setScheduleKindLoc(Reader->ReadSourceLocation());
  C->setCommaLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPOrderedClause(OMPOrderedClause *C) {
  C->setNumForLoops(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPNowaitClause(OMPNowaitClause *) {}

void OMPClauseReader::VisitOMPUntiedClause(OMPUntiedClause *) {}

void OMPClauseReader::VisitOMPMergeableClause(OMPMergeableClause *) {}

void OMPClauseReader::VisitOMPReadClause(OMPReadClause *) {}

void OMPClauseReader::VisitOMPWriteClause(OMPWriteClause *) {}

void OMPClauseReader::VisitOMPUpdateClause(OMPUpdateClause *) {}

void OMPClauseReader::VisitOMPCaptureClause(OMPCaptureClause *) {}

void OMPClauseReader::VisitOMPSeqCstClause(OMPSeqCstClause *) {}

void OMPClauseReader::VisitOMPThreadsClause(OMPThreadsClause *) {}

void OMPClauseReader::VisitOMPSIMDClause(OMPSIMDClause *) {}

void OMPClauseReader::VisitOMPNogroupClause(OMPNogroupClause *) {}

void OMPClauseReader::VisitOMPPrivateClause(OMPPrivateClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setPrivateCopies(Vars);
}

void OMPClauseReader::VisitOMPFirstprivateClause(OMPFirstprivateClause *C) {
  VisitOMPClauseWithPreInit(C);
  C->setLParenLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setPrivateCopies(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setInits(Vars);
}

void OMPClauseReader::VisitOMPLastprivateClause(OMPLastprivateClause *C) {
  VisitOMPClauseWithPostUpdate(C);
  C->setLParenLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setPrivateCopies(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setSourceExprs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setDestinationExprs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setAssignmentOps(Vars);
}

void OMPClauseReader::VisitOMPSharedClause(OMPSharedClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
}

void OMPClauseReader::VisitOMPReductionClause(OMPReductionClause *C) {
  VisitOMPClauseWithPostUpdate(C);
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setColonLoc(Reader->ReadSourceLocation());
  NestedNameSpecifierLoc NNSL = Reader->Record.ReadNestedNameSpecifierLoc(Idx);
  DeclarationNameInfo DNI;
  Reader->ReadDeclarationNameInfo(DNI);
  C->setQualifierLoc(NNSL);
  C->setNameInfo(DNI);

  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setPrivates(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setLHSExprs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setRHSExprs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setReductionOps(Vars);
}

void OMPClauseReader::VisitOMPLinearClause(OMPLinearClause *C) {
  VisitOMPClauseWithPostUpdate(C);
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setColonLoc(Reader->ReadSourceLocation());
  C->setModifier(static_cast<OpenMPLinearClauseKind>(Reader->Record[Idx++]));
  C->setModifierLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setPrivates(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setInits(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setUpdates(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setFinals(Vars);
  C->setStep(Reader->Record.ReadSubExpr());
  C->setCalcStep(Reader->Record.ReadSubExpr());
}

void OMPClauseReader::VisitOMPAlignedClause(OMPAlignedClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setColonLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
  C->setAlignment(Reader->Record.ReadSubExpr());
}

void OMPClauseReader::VisitOMPCopyinClause(OMPCopyinClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Exprs;
  Exprs.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Exprs.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Exprs);
  Exprs.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Exprs.push_back(Reader->Record.ReadSubExpr());
  C->setSourceExprs(Exprs);
  Exprs.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Exprs.push_back(Reader->Record.ReadSubExpr());
  C->setDestinationExprs(Exprs);
  Exprs.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Exprs.push_back(Reader->Record.ReadSubExpr());
  C->setAssignmentOps(Exprs);
}

void OMPClauseReader::VisitOMPCopyprivateClause(OMPCopyprivateClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Exprs;
  Exprs.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Exprs.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Exprs);
  Exprs.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Exprs.push_back(Reader->Record.ReadSubExpr());
  C->setSourceExprs(Exprs);
  Exprs.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Exprs.push_back(Reader->Record.ReadSubExpr());
  C->setDestinationExprs(Exprs);
  Exprs.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Exprs.push_back(Reader->Record.ReadSubExpr());
  C->setAssignmentOps(Exprs);
}

void OMPClauseReader::VisitOMPFlushClause(OMPFlushClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
}

void OMPClauseReader::VisitOMPDependClause(OMPDependClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setDependencyKind(
      static_cast<OpenMPDependClauseKind>(Reader->Record[Idx++]));
  C->setDependencyLoc(Reader->ReadSourceLocation());
  C->setColonLoc(Reader->ReadSourceLocation());
  unsigned NumVars = C->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
  C->setCounterValue(Reader->Record.ReadSubExpr());
}

void OMPClauseReader::VisitOMPDeviceClause(OMPDeviceClause *C) {
  C->setDevice(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPMapClause(OMPMapClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setMapTypeModifier(
     static_cast<OpenMPMapClauseKind>(Reader->Record[Idx++]));
  C->setMapType(
     static_cast<OpenMPMapClauseKind>(Reader->Record[Idx++]));
  C->setMapLoc(Reader->ReadSourceLocation());
  C->setColonLoc(Reader->ReadSourceLocation());
  auto NumVars = C->varlist_size();
  auto UniqueDecls = C->getUniqueDeclarationsNum();
  auto TotalLists = C->getTotalComponentListNum();
  auto TotalComponents = C->getTotalComponentsNum();

  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);

  SmallVector<ValueDecl *, 16> Decls;
  Decls.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    Decls.push_back(Reader->Record.ReadDeclAs<ValueDecl>(Idx));
  C->setUniqueDecls(Decls);

  SmallVector<unsigned, 16> ListsPerDecl;
  ListsPerDecl.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    ListsPerDecl.push_back(Reader->Record[Idx++]);
  C->setDeclNumLists(ListsPerDecl);

  SmallVector<unsigned, 32> ListSizes;
  ListSizes.reserve(TotalLists);
  for (unsigned i = 0; i < TotalLists; ++i)
    ListSizes.push_back(Reader->Record[Idx++]);
  C->setComponentListSizes(ListSizes);

  SmallVector<OMPClauseMappableExprCommon::MappableComponent, 32> Components;
  Components.reserve(TotalComponents);
  for (unsigned i = 0; i < TotalComponents; ++i) {
    Expr *AssociatedExpr = Reader->Record.ReadSubExpr();
    ValueDecl *AssociatedDecl = Reader->Record.ReadDeclAs<ValueDecl>(Idx);
    Components.push_back(OMPClauseMappableExprCommon::MappableComponent(
        AssociatedExpr, AssociatedDecl));
  }
  C->setComponents(Components, ListSizes);
}

void OMPClauseReader::VisitOMPNumTeamsClause(OMPNumTeamsClause *C) {
  C->setNumTeams(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPThreadLimitClause(OMPThreadLimitClause *C) {
  C->setThreadLimit(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPPriorityClause(OMPPriorityClause *C) {
  C->setPriority(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPGrainsizeClause(OMPGrainsizeClause *C) {
  C->setGrainsize(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPNumTasksClause(OMPNumTasksClause *C) {
  C->setNumTasks(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPHintClause(OMPHintClause *C) {
  C->setHint(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPDistScheduleClause(OMPDistScheduleClause *C) {
  VisitOMPClauseWithPreInit(C);
  C->setDistScheduleKind(
      static_cast<OpenMPDistScheduleClauseKind>(Reader->Record[Idx++]));
  C->setChunkSize(Reader->Record.ReadSubExpr());
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setDistScheduleKindLoc(Reader->ReadSourceLocation());
  C->setCommaLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPDefaultmapClause(OMPDefaultmapClause *C) {
  C->setDefaultmapKind(
       static_cast<OpenMPDefaultmapClauseKind>(Reader->Record[Idx++]));
  C->setDefaultmapModifier(
      static_cast<OpenMPDefaultmapClauseModifier>(Reader->Record[Idx++]));
  C->setLParenLoc(Reader->ReadSourceLocation());
  C->setDefaultmapModifierLoc(Reader->ReadSourceLocation());
  C->setDefaultmapKindLoc(Reader->ReadSourceLocation());
}

void OMPClauseReader::VisitOMPToClause(OMPToClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  auto NumVars = C->varlist_size();
  auto UniqueDecls = C->getUniqueDeclarationsNum();
  auto TotalLists = C->getTotalComponentListNum();
  auto TotalComponents = C->getTotalComponentsNum();

  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);

  SmallVector<ValueDecl *, 16> Decls;
  Decls.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    Decls.push_back(Reader->Record.ReadDeclAs<ValueDecl>(Idx));
  C->setUniqueDecls(Decls);

  SmallVector<unsigned, 16> ListsPerDecl;
  ListsPerDecl.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    ListsPerDecl.push_back(Reader->Record[Idx++]);
  C->setDeclNumLists(ListsPerDecl);

  SmallVector<unsigned, 32> ListSizes;
  ListSizes.reserve(TotalLists);
  for (unsigned i = 0; i < TotalLists; ++i)
    ListSizes.push_back(Reader->Record[Idx++]);
  C->setComponentListSizes(ListSizes);

  SmallVector<OMPClauseMappableExprCommon::MappableComponent, 32> Components;
  Components.reserve(TotalComponents);
  for (unsigned i = 0; i < TotalComponents; ++i) {
    Expr *AssociatedExpr = Reader->Record.ReadSubExpr();
    ValueDecl *AssociatedDecl = Reader->Record.ReadDeclAs<ValueDecl>(Idx);
    Components.push_back(OMPClauseMappableExprCommon::MappableComponent(
        AssociatedExpr, AssociatedDecl));
  }
  C->setComponents(Components, ListSizes);
}

void OMPClauseReader::VisitOMPFromClause(OMPFromClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  auto NumVars = C->varlist_size();
  auto UniqueDecls = C->getUniqueDeclarationsNum();
  auto TotalLists = C->getTotalComponentListNum();
  auto TotalComponents = C->getTotalComponentsNum();

  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);

  SmallVector<ValueDecl *, 16> Decls;
  Decls.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    Decls.push_back(Reader->Record.ReadDeclAs<ValueDecl>(Idx));
  C->setUniqueDecls(Decls);

  SmallVector<unsigned, 16> ListsPerDecl;
  ListsPerDecl.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    ListsPerDecl.push_back(Reader->Record[Idx++]);
  C->setDeclNumLists(ListsPerDecl);

  SmallVector<unsigned, 32> ListSizes;
  ListSizes.reserve(TotalLists);
  for (unsigned i = 0; i < TotalLists; ++i)
    ListSizes.push_back(Reader->Record[Idx++]);
  C->setComponentListSizes(ListSizes);

  SmallVector<OMPClauseMappableExprCommon::MappableComponent, 32> Components;
  Components.reserve(TotalComponents);
  for (unsigned i = 0; i < TotalComponents; ++i) {
    Expr *AssociatedExpr = Reader->Record.ReadSubExpr();
    ValueDecl *AssociatedDecl = Reader->Record.ReadDeclAs<ValueDecl>(Idx);
    Components.push_back(OMPClauseMappableExprCommon::MappableComponent(
        AssociatedExpr, AssociatedDecl));
  }
  C->setComponents(Components, ListSizes);
}

void OMPClauseReader::VisitOMPUseDevicePtrClause(OMPUseDevicePtrClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  auto NumVars = C->varlist_size();
  auto UniqueDecls = C->getUniqueDeclarationsNum();
  auto TotalLists = C->getTotalComponentListNum();
  auto TotalComponents = C->getTotalComponentsNum();

  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setPrivateCopies(Vars);
  Vars.clear();
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setInits(Vars);

  SmallVector<ValueDecl *, 16> Decls;
  Decls.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    Decls.push_back(Reader->Record.ReadDeclAs<ValueDecl>(Idx));
  C->setUniqueDecls(Decls);

  SmallVector<unsigned, 16> ListsPerDecl;
  ListsPerDecl.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    ListsPerDecl.push_back(Reader->Record[Idx++]);
  C->setDeclNumLists(ListsPerDecl);

  SmallVector<unsigned, 32> ListSizes;
  ListSizes.reserve(TotalLists);
  for (unsigned i = 0; i < TotalLists; ++i)
    ListSizes.push_back(Reader->Record[Idx++]);
  C->setComponentListSizes(ListSizes);

  SmallVector<OMPClauseMappableExprCommon::MappableComponent, 32> Components;
  Components.reserve(TotalComponents);
  for (unsigned i = 0; i < TotalComponents; ++i) {
    Expr *AssociatedExpr = Reader->Record.ReadSubExpr();
    ValueDecl *AssociatedDecl = Reader->Record.ReadDeclAs<ValueDecl>(Idx);
    Components.push_back(OMPClauseMappableExprCommon::MappableComponent(
        AssociatedExpr, AssociatedDecl));
  }
  C->setComponents(Components, ListSizes);
}

void OMPClauseReader::VisitOMPIsDevicePtrClause(OMPIsDevicePtrClause *C) {
  C->setLParenLoc(Reader->ReadSourceLocation());
  auto NumVars = C->varlist_size();
  auto UniqueDecls = C->getUniqueDeclarationsNum();
  auto TotalLists = C->getTotalComponentListNum();
  auto TotalComponents = C->getTotalComponentsNum();

  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i)
    Vars.push_back(Reader->Record.ReadSubExpr());
  C->setVarRefs(Vars);
  Vars.clear();

  SmallVector<ValueDecl *, 16> Decls;
  Decls.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    Decls.push_back(Reader->Record.ReadDeclAs<ValueDecl>(Idx));
  C->setUniqueDecls(Decls);

  SmallVector<unsigned, 16> ListsPerDecl;
  ListsPerDecl.reserve(UniqueDecls);
  for (unsigned i = 0; i < UniqueDecls; ++i)
    ListsPerDecl.push_back(Reader->Record[Idx++]);
  C->setDeclNumLists(ListsPerDecl);

  SmallVector<unsigned, 32> ListSizes;
  ListSizes.reserve(TotalLists);
  for (unsigned i = 0; i < TotalLists; ++i)
    ListSizes.push_back(Reader->Record[Idx++]);
  C->setComponentListSizes(ListSizes);

  SmallVector<OMPClauseMappableExprCommon::MappableComponent, 32> Components;
  Components.reserve(TotalComponents);
  for (unsigned i = 0; i < TotalComponents; ++i) {
    Expr *AssociatedExpr = Reader->Record.ReadSubExpr();
    ValueDecl *AssociatedDecl = Reader->Record.ReadDeclAs<ValueDecl>(Idx);
    Components.push_back(OMPClauseMappableExprCommon::MappableComponent(
        AssociatedExpr, AssociatedDecl));
  }
  C->setComponents(Components, ListSizes);
}

//===----------------------------------------------------------------------===//
// OpenMP Directives.
//===----------------------------------------------------------------------===//
void ASTStmtReader::VisitOMPExecutableDirective(OMPExecutableDirective *E) {
  E->setLocStart(ReadSourceLocation());
  E->setLocEnd(ReadSourceLocation());
  OMPClauseReader ClauseReader(this, Record, Idx);
  SmallVector<OMPClause *, 5> Clauses;
  for (unsigned i = 0; i < E->getNumClauses(); ++i)
    Clauses.push_back(ClauseReader.readClause());
  E->setClauses(Clauses);
  if (E->hasAssociatedStmt())
    E->setAssociatedStmt(Record.ReadSubStmt());
}

void ASTStmtReader::VisitOMPLoopDirective(OMPLoopDirective *D) {
  VisitStmt(D);
  // Two fields (NumClauses and CollapsedNum) were read in ReadStmtFromStream.
  Idx += 2;
  VisitOMPExecutableDirective(D);
  D->setIterationVariable(Record.ReadSubExpr());
  D->setLastIteration(Record.ReadSubExpr());
  D->setCalcLastIteration(Record.ReadSubExpr());
  D->setPreCond(Record.ReadSubExpr());
  D->setCond(Record.ReadSubExpr());
  D->setInit(Record.ReadSubExpr());
  D->setInc(Record.ReadSubExpr());
  D->setPreInits(Record.ReadSubStmt());
  if (isOpenMPWorksharingDirective(D->getDirectiveKind()) ||
      isOpenMPTaskLoopDirective(D->getDirectiveKind()) ||
      isOpenMPDistributeDirective(D->getDirectiveKind())) {
    D->setIsLastIterVariable(Record.ReadSubExpr());
    D->setLowerBoundVariable(Record.ReadSubExpr());
    D->setUpperBoundVariable(Record.ReadSubExpr());
    D->setStrideVariable(Record.ReadSubExpr());
    D->setEnsureUpperBound(Record.ReadSubExpr());
    D->setNextLowerBound(Record.ReadSubExpr());
    D->setNextUpperBound(Record.ReadSubExpr());
    D->setNumIterations(Record.ReadSubExpr());
  }
  if (isOpenMPLoopBoundSharingDirective(D->getDirectiveKind())) {
    D->setPrevLowerBoundVariable(Record.ReadSubExpr());
    D->setPrevUpperBoundVariable(Record.ReadSubExpr());
  }
  SmallVector<Expr *, 4> Sub;
  unsigned CollapsedNum = D->getCollapsedNumber();
  Sub.reserve(CollapsedNum);
  for (unsigned i = 0; i < CollapsedNum; ++i)
    Sub.push_back(Record.ReadSubExpr());
  D->setCounters(Sub);
  Sub.clear();
  for (unsigned i = 0; i < CollapsedNum; ++i)
    Sub.push_back(Record.ReadSubExpr());
  D->setPrivateCounters(Sub);
  Sub.clear();
  for (unsigned i = 0; i < CollapsedNum; ++i)
    Sub.push_back(Record.ReadSubExpr());
  D->setInits(Sub);
  Sub.clear();
  for (unsigned i = 0; i < CollapsedNum; ++i)
    Sub.push_back(Record.ReadSubExpr());
  D->setUpdates(Sub);
  Sub.clear();
  for (unsigned i = 0; i < CollapsedNum; ++i)
    Sub.push_back(Record.ReadSubExpr());
  D->setFinals(Sub);
}

void ASTStmtReader::VisitOMPParallelDirective(OMPParallelDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
  D->setHasCancel(Record[Idx++]);
}

void ASTStmtReader::VisitOMPSimdDirective(OMPSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPForDirective(OMPForDirective *D) {
  VisitOMPLoopDirective(D);
  D->setHasCancel(Record[Idx++]);
}

void ASTStmtReader::VisitOMPForSimdDirective(OMPForSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPSectionsDirective(OMPSectionsDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
  D->setHasCancel(Record[Idx++]);
}

void ASTStmtReader::VisitOMPSectionDirective(OMPSectionDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
  D->setHasCancel(Record[Idx++]);
}

void ASTStmtReader::VisitOMPSingleDirective(OMPSingleDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPMasterDirective(OMPMasterDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPCriticalDirective(OMPCriticalDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
  ReadDeclarationNameInfo(D->DirName);
}

void ASTStmtReader::VisitOMPParallelForDirective(OMPParallelForDirective *D) {
  VisitOMPLoopDirective(D);
  D->setHasCancel(Record[Idx++]);
}

void ASTStmtReader::VisitOMPParallelForSimdDirective(
    OMPParallelForSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPParallelSectionsDirective(
    OMPParallelSectionsDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
  D->setHasCancel(Record[Idx++]);
}

void ASTStmtReader::VisitOMPTaskDirective(OMPTaskDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
  D->setHasCancel(Record[Idx++]);
}

void ASTStmtReader::VisitOMPTaskyieldDirective(OMPTaskyieldDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPBarrierDirective(OMPBarrierDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPTaskwaitDirective(OMPTaskwaitDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPTaskgroupDirective(OMPTaskgroupDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPFlushDirective(OMPFlushDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPOrderedDirective(OMPOrderedDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPAtomicDirective(OMPAtomicDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
  D->setX(Record.ReadSubExpr());
  D->setV(Record.ReadSubExpr());
  D->setExpr(Record.ReadSubExpr());
  D->setUpdateExpr(Record.ReadSubExpr());
  D->IsXLHSInRHSPart = Record[Idx++] != 0;
  D->IsPostfixUpdate = Record[Idx++] != 0;
}

void ASTStmtReader::VisitOMPTargetDirective(OMPTargetDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPTargetDataDirective(OMPTargetDataDirective *D) {
  VisitStmt(D);
  ++Idx;
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPTargetEnterDataDirective(
    OMPTargetEnterDataDirective *D) {
  VisitStmt(D);
  ++Idx;
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPTargetExitDataDirective(
    OMPTargetExitDataDirective *D) {
  VisitStmt(D);
  ++Idx;
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPTargetParallelDirective(
    OMPTargetParallelDirective *D) {
  VisitStmt(D);
  ++Idx;
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPTargetParallelForDirective(
    OMPTargetParallelForDirective *D) {
  VisitOMPLoopDirective(D);
  D->setHasCancel(Record[Idx++]);
}

void ASTStmtReader::VisitOMPTeamsDirective(OMPTeamsDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
}

void ASTStmtReader::VisitOMPCancellationPointDirective(
    OMPCancellationPointDirective *D) {
  VisitStmt(D);
  VisitOMPExecutableDirective(D);
  D->setCancelRegion(static_cast<OpenMPDirectiveKind>(Record[Idx++]));
}

void ASTStmtReader::VisitOMPCancelDirective(OMPCancelDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
  D->setCancelRegion(static_cast<OpenMPDirectiveKind>(Record[Idx++]));
}

void ASTStmtReader::VisitOMPTaskLoopDirective(OMPTaskLoopDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPTaskLoopSimdDirective(OMPTaskLoopSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPDistributeDirective(OMPDistributeDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPTargetUpdateDirective(OMPTargetUpdateDirective *D) {
  VisitStmt(D);
  ++Idx;
  VisitOMPExecutableDirective(D);
}
void ASTStmtReader::VisitOMPDistributeParallelForDirective(
    OMPDistributeParallelForDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPDistributeParallelForSimdDirective(
    OMPDistributeParallelForSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPDistributeSimdDirective(
    OMPDistributeSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPTargetParallelForSimdDirective(
    OMPTargetParallelForSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPTargetSimdDirective(OMPTargetSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPTeamsDistributeDirective(
    OMPTeamsDistributeDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPTeamsDistributeSimdDirective(
    OMPTeamsDistributeSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPTeamsDistributeParallelForSimdDirective(
    OMPTeamsDistributeParallelForSimdDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPTeamsDistributeParallelForDirective(
    OMPTeamsDistributeParallelForDirective *D) {
  VisitOMPLoopDirective(D);
}

void ASTStmtReader::VisitOMPTargetTeamsDirective(OMPTargetTeamsDirective *D) {
  VisitStmt(D);
  // The NumClauses field was read in ReadStmtFromStream.
  ++Idx;
  VisitOMPExecutableDirective(D);
}

//===----------------------------------------------------------------------===//
// ASTReader Implementation
//===----------------------------------------------------------------------===//

Stmt *ASTReader::ReadStmt(ModuleFile &F) {
  switch (ReadingKind) {
  case Read_None:
    llvm_unreachable("should not call this when not reading anything");
  case Read_Decl:
  case Read_Type:
    return ReadStmtFromStream(F);
  case Read_Stmt:
    return ReadSubStmt();
  }

  llvm_unreachable("ReadingKind not set ?");
}

Expr *ASTReader::ReadExpr(ModuleFile &F) {
  return cast_or_null<Expr>(ReadStmt(F));
}

Expr *ASTReader::ReadSubExpr() {
  return cast_or_null<Expr>(ReadSubStmt());
}

// Within the bitstream, expressions are stored in Reverse Polish
// Notation, with each of the subexpressions preceding the
// expression they are stored in. Subexpressions are stored from last to first.
// To evaluate expressions, we continue reading expressions and placing them on
// the stack, with expressions having operands removing those operands from the
// stack. Evaluation terminates when we see a STMT_STOP record, and
// the single remaining expression on the stack is our result.
Stmt *ASTReader::ReadStmtFromStream(ModuleFile &F) {

  ReadingKindTracker ReadingKind(Read_Stmt, *this);
  llvm::BitstreamCursor &Cursor = F.DeclsCursor;

  // Map of offset to previously deserialized stmt. The offset points
  /// just after the stmt record.
  llvm::DenseMap<uint64_t, Stmt *> StmtEntries;

#ifndef NDEBUG
  unsigned PrevNumStmts = StmtStack.size();
#endif

  RecordData Record;
  unsigned Idx;
  ASTStmtReader Reader(*this, F, Cursor, Record, Idx);
  Stmt::EmptyShell Empty;

  while (true) {
    llvm::BitstreamEntry Entry = Cursor.advanceSkippingSubblocks();

    switch (Entry.Kind) {
    case llvm::BitstreamEntry::SubBlock: // Handled for us already.
    case llvm::BitstreamEntry::Error:
      Error("malformed block record in AST file");
      return nullptr;
    case llvm::BitstreamEntry::EndBlock:
      goto Done;
    case llvm::BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    Stmt *S = nullptr;
    Idx = 0;
    Record.clear();
    bool Finished = false;
    bool IsStmtReference = false;
    switch ((StmtCode)Cursor.readRecord(Entry.ID, Record)) {
    case STMT_STOP:
      Finished = true;
      break;

    case STMT_REF_PTR:
      IsStmtReference = true;
      assert(StmtEntries.find(Record[0]) != StmtEntries.end() &&
             "No stmt was recorded for this offset reference!");
      S = StmtEntries[Record[Idx++]];
      break;

    case STMT_NULL_PTR:
      S = nullptr;
      break;

    case STMT_NULL:
      S = new (Context) NullStmt(Empty);
      break;

    case STMT_COMPOUND:
      S = new (Context) CompoundStmt(Empty);
      break;

    case STMT_CASE:
      S = new (Context) CaseStmt(Empty);
      break;

    case STMT_DEFAULT:
      S = new (Context) DefaultStmt(Empty);
      break;

    case STMT_LABEL:
      S = new (Context) LabelStmt(Empty);
      break;

    case STMT_ATTRIBUTED:
      S = AttributedStmt::CreateEmpty(
        Context,
        /*NumAttrs*/Record[ASTStmtReader::NumStmtFields]);
      break;

    case STMT_IF:
      S = new (Context) IfStmt(Empty);
      break;

    case STMT_SWITCH:
      S = new (Context) SwitchStmt(Empty);
      break;

    case STMT_WHILE:
      S = new (Context) WhileStmt(Empty);
      break;

    case STMT_DO:
      S = new (Context) DoStmt(Empty);
      break;

    case STMT_FOR:
      S = new (Context) ForStmt(Empty);
      break;

    case STMT_GOTO:
      S = new (Context) GotoStmt(Empty);
      break;

    case STMT_INDIRECT_GOTO:
      S = new (Context) IndirectGotoStmt(Empty);
      break;

    case STMT_CONTINUE:
      S = new (Context) ContinueStmt(Empty);
      break;

    case STMT_BREAK:
      S = new (Context) BreakStmt(Empty);
      break;

    case STMT_RETURN:
      S = new (Context) ReturnStmt(Empty);
      break;

    case STMT_DECL:
      S = new (Context) DeclStmt(Empty);
      break;

    case STMT_GCCASM:
      S = new (Context) GCCAsmStmt(Empty);
      break;

    case STMT_MSASM:
      S = new (Context) MSAsmStmt(Empty);
      break;

    case STMT_CAPTURED:
      S = CapturedStmt::CreateDeserialized(Context,
                                           Record[ASTStmtReader::NumStmtFields]);
      break;

    case EXPR_PREDEFINED:
      S = new (Context) PredefinedExpr(Empty);
      break;

    case EXPR_DECL_REF:
      S = DeclRefExpr::CreateEmpty(
        Context,
        /*HasQualifier=*/Record[ASTStmtReader::NumExprFields],
        /*HasFoundDecl=*/Record[ASTStmtReader::NumExprFields + 1],
        /*HasTemplateKWAndArgsInfo=*/Record[ASTStmtReader::NumExprFields + 2],
        /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields + 2] ?
          Record[ASTStmtReader::NumExprFields + 5] : 0);
      break;

    case EXPR_INTEGER_LITERAL:
      S = IntegerLiteral::Create(Context, Empty);
      break;

    case EXPR_FLOATING_LITERAL:
      S = FloatingLiteral::Create(Context, Empty);
      break;

    case EXPR_IMAGINARY_LITERAL:
      S = new (Context) ImaginaryLiteral(Empty);
      break;

    case EXPR_STRING_LITERAL:
      S = StringLiteral::CreateEmpty(Context,
                                     Record[ASTStmtReader::NumExprFields + 1]);
      break;

    case EXPR_CHARACTER_LITERAL:
      S = new (Context) CharacterLiteral(Empty);
      break;

    case EXPR_PAREN:
      S = new (Context) ParenExpr(Empty);
      break;

    case EXPR_PAREN_LIST:
      S = new (Context) ParenListExpr(Empty);
      break;

    case EXPR_UNARY_OPERATOR:
      S = new (Context) UnaryOperator(Empty);
      break;

    case EXPR_OFFSETOF:
      S = OffsetOfExpr::CreateEmpty(Context,
                                    Record[ASTStmtReader::NumExprFields],
                                    Record[ASTStmtReader::NumExprFields + 1]);
      break;

    case EXPR_SIZEOF_ALIGN_OF:
      S = new (Context) UnaryExprOrTypeTraitExpr(Empty);
      break;

    case EXPR_ARRAY_SUBSCRIPT:
      S = new (Context) ArraySubscriptExpr(Empty);
      break;

    case EXPR_OMP_ARRAY_SECTION:
      S = new (Context) OMPArraySectionExpr(Empty);
      break;

    case EXPR_CALL:
      S = new (Context) CallExpr(Context, Stmt::CallExprClass, Empty);
      break;

    case EXPR_MEMBER: {
      // We load everything here and fully initialize it at creation.
      // That way we can use MemberExpr::Create and don't have to duplicate its
      // logic with a MemberExpr::CreateEmpty.

      assert(Idx == 0);
      NestedNameSpecifierLoc QualifierLoc;
      if (Record[Idx++]) { // HasQualifier.
        QualifierLoc = ReadNestedNameSpecifierLoc(F, Record, Idx);
      }

      SourceLocation TemplateKWLoc;
      TemplateArgumentListInfo ArgInfo;
      bool HasTemplateKWAndArgsInfo = Record[Idx++];
      if (HasTemplateKWAndArgsInfo) {
        TemplateKWLoc = ReadSourceLocation(F, Record, Idx);
        unsigned NumTemplateArgs = Record[Idx++];
        ArgInfo.setLAngleLoc(ReadSourceLocation(F, Record, Idx));
        ArgInfo.setRAngleLoc(ReadSourceLocation(F, Record, Idx));
        for (unsigned i = 0; i != NumTemplateArgs; ++i)
          ArgInfo.addArgument(ReadTemplateArgumentLoc(F, Record, Idx));
      }

      bool HadMultipleCandidates = Record[Idx++];

      NamedDecl *FoundD = ReadDeclAs<NamedDecl>(F, Record, Idx);
      AccessSpecifier AS = (AccessSpecifier)Record[Idx++];
      DeclAccessPair FoundDecl = DeclAccessPair::make(FoundD, AS);

      QualType T = readType(F, Record, Idx);
      ExprValueKind VK = static_cast<ExprValueKind>(Record[Idx++]);
      ExprObjectKind OK = static_cast<ExprObjectKind>(Record[Idx++]);
      Expr *Base = ReadSubExpr();
      ValueDecl *MemberD = ReadDeclAs<ValueDecl>(F, Record, Idx);
      SourceLocation MemberLoc = ReadSourceLocation(F, Record, Idx);
      DeclarationNameInfo MemberNameInfo(MemberD->getDeclName(), MemberLoc);
      bool IsArrow = Record[Idx++];
      SourceLocation OperatorLoc = ReadSourceLocation(F, Record, Idx);

      S = MemberExpr::Create(Context, Base, IsArrow, OperatorLoc, QualifierLoc,
                             TemplateKWLoc, MemberD, FoundDecl, MemberNameInfo,
                             HasTemplateKWAndArgsInfo ? &ArgInfo : nullptr, T,
                             VK, OK);
      ReadDeclarationNameLoc(F, cast<MemberExpr>(S)->MemberDNLoc,
                             MemberD->getDeclName(), Record, Idx);
      if (HadMultipleCandidates)
        cast<MemberExpr>(S)->setHadMultipleCandidates(true);
      break;
    }

    case EXPR_BINARY_OPERATOR:
      S = new (Context) BinaryOperator(Empty);
      break;

    case EXPR_COMPOUND_ASSIGN_OPERATOR:
      S = new (Context) CompoundAssignOperator(Empty);
      break;

    case EXPR_CONDITIONAL_OPERATOR:
      S = new (Context) ConditionalOperator(Empty);
      break;

    case EXPR_BINARY_CONDITIONAL_OPERATOR:
      S = new (Context) BinaryConditionalOperator(Empty);
      break;

    case EXPR_IMPLICIT_CAST:
      S = ImplicitCastExpr::CreateEmpty(Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CSTYLE_CAST:
      S = CStyleCastExpr::CreateEmpty(Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_COMPOUND_LITERAL:
      S = new (Context) CompoundLiteralExpr(Empty);
      break;

    case EXPR_EXT_VECTOR_ELEMENT:
      S = new (Context) ExtVectorElementExpr(Empty);
      break;

    case EXPR_INIT_LIST:
      S = new (Context) InitListExpr(Empty);
      break;

    case EXPR_DESIGNATED_INIT:
      S = DesignatedInitExpr::CreateEmpty(Context,
                                     Record[ASTStmtReader::NumExprFields] - 1);

      break;

    case EXPR_DESIGNATED_INIT_UPDATE:
      S = new (Context) DesignatedInitUpdateExpr(Empty);
      break;

    case EXPR_IMPLICIT_VALUE_INIT:
      S = new (Context) ImplicitValueInitExpr(Empty);
      break;

    case EXPR_NO_INIT:
      S = new (Context) NoInitExpr(Empty);
      break;

    case EXPR_ARRAY_INIT_LOOP:
      S = new (Context) ArrayInitLoopExpr(Empty);
      break;

    case EXPR_ARRAY_INIT_INDEX:
      S = new (Context) ArrayInitIndexExpr(Empty);
      break;

    case EXPR_VA_ARG:
      S = new (Context) VAArgExpr(Empty);
      break;

    case EXPR_ADDR_LABEL:
      S = new (Context) AddrLabelExpr(Empty);
      break;

    case EXPR_STMT:
      S = new (Context) StmtExpr(Empty);
      break;

    case EXPR_CHOOSE:
      S = new (Context) ChooseExpr(Empty);
      break;

    case EXPR_GNU_NULL:
      S = new (Context) GNUNullExpr(Empty);
      break;

    case EXPR_SHUFFLE_VECTOR:
      S = new (Context) ShuffleVectorExpr(Empty);
      break;

    case EXPR_CONVERT_VECTOR:
      S = new (Context) ConvertVectorExpr(Empty);
      break;

    case EXPR_BLOCK:
      S = new (Context) BlockExpr(Empty);
      break;

    case EXPR_GENERIC_SELECTION:
      S = new (Context) GenericSelectionExpr(Empty);
      break;

    case EXPR_OBJC_STRING_LITERAL:
      S = new (Context) ObjCStringLiteral(Empty);
      break;
    case EXPR_OBJC_BOXED_EXPRESSION:
      S = new (Context) ObjCBoxedExpr(Empty);
      break;
    case EXPR_OBJC_ARRAY_LITERAL:
      S = ObjCArrayLiteral::CreateEmpty(Context,
                                        Record[ASTStmtReader::NumExprFields]);
      break;
    case EXPR_OBJC_DICTIONARY_LITERAL:
      S = ObjCDictionaryLiteral::CreateEmpty(Context,
            Record[ASTStmtReader::NumExprFields],
            Record[ASTStmtReader::NumExprFields + 1]);
      break;
    case EXPR_OBJC_ENCODE:
      S = new (Context) ObjCEncodeExpr(Empty);
      break;
    case EXPR_OBJC_SELECTOR_EXPR:
      S = new (Context) ObjCSelectorExpr(Empty);
      break;
    case EXPR_OBJC_PROTOCOL_EXPR:
      S = new (Context) ObjCProtocolExpr(Empty);
      break;
    case EXPR_OBJC_IVAR_REF_EXPR:
      S = new (Context) ObjCIvarRefExpr(Empty);
      break;
    case EXPR_OBJC_PROPERTY_REF_EXPR:
      S = new (Context) ObjCPropertyRefExpr(Empty);
      break;
    case EXPR_OBJC_SUBSCRIPT_REF_EXPR:
      S = new (Context) ObjCSubscriptRefExpr(Empty);
      break;
    case EXPR_OBJC_KVC_REF_EXPR:
      llvm_unreachable("mismatching AST file");
    case EXPR_OBJC_MESSAGE_EXPR:
      S = ObjCMessageExpr::CreateEmpty(Context,
                                     Record[ASTStmtReader::NumExprFields],
                                     Record[ASTStmtReader::NumExprFields + 1]);
      break;
    case EXPR_OBJC_ISA:
      S = new (Context) ObjCIsaExpr(Empty);
      break;
    case EXPR_OBJC_INDIRECT_COPY_RESTORE:
      S = new (Context) ObjCIndirectCopyRestoreExpr(Empty);
      break;
    case EXPR_OBJC_BRIDGED_CAST:
      S = new (Context) ObjCBridgedCastExpr(Empty);
      break;
    case STMT_OBJC_FOR_COLLECTION:
      S = new (Context) ObjCForCollectionStmt(Empty);
      break;
    case STMT_OBJC_CATCH:
      S = new (Context) ObjCAtCatchStmt(Empty);
      break;
    case STMT_OBJC_FINALLY:
      S = new (Context) ObjCAtFinallyStmt(Empty);
      break;
    case STMT_OBJC_AT_TRY:
      S = ObjCAtTryStmt::CreateEmpty(Context,
                                     Record[ASTStmtReader::NumStmtFields],
                                     Record[ASTStmtReader::NumStmtFields + 1]);
      break;
    case STMT_OBJC_AT_SYNCHRONIZED:
      S = new (Context) ObjCAtSynchronizedStmt(Empty);
      break;
    case STMT_OBJC_AT_THROW:
      S = new (Context) ObjCAtThrowStmt(Empty);
      break;
    case STMT_OBJC_AUTORELEASE_POOL:
      S = new (Context) ObjCAutoreleasePoolStmt(Empty);
      break;
    case EXPR_OBJC_BOOL_LITERAL:
      S = new (Context) ObjCBoolLiteralExpr(Empty);
      break;
    case EXPR_OBJC_AVAILABILITY_CHECK:
      S = new (Context) ObjCAvailabilityCheckExpr(Empty);
      break;
    case STMT_SEH_LEAVE:
      S = new (Context) SEHLeaveStmt(Empty);
      break;
    case STMT_SEH_EXCEPT:
      S = new (Context) SEHExceptStmt(Empty);
      break;
    case STMT_SEH_FINALLY:
      S = new (Context) SEHFinallyStmt(Empty);
      break;
    case STMT_SEH_TRY:
      S = new (Context) SEHTryStmt(Empty);
      break;
    case STMT_CXX_CATCH:
      S = new (Context) CXXCatchStmt(Empty);
      break;

    case STMT_CXX_TRY:
      S = CXXTryStmt::Create(Context, Empty,
             /*NumHandlers=*/Record[ASTStmtReader::NumStmtFields]);
      break;

    case STMT_CXX_FOR_RANGE:
      S = new (Context) CXXForRangeStmt(Empty);
      break;

    case STMT_MS_DEPENDENT_EXISTS:
      S = new (Context) MSDependentExistsStmt(SourceLocation(), true,
                                              NestedNameSpecifierLoc(),
                                              DeclarationNameInfo(),
                                              nullptr);
      break;

    case STMT_OMP_PARALLEL_DIRECTIVE:
      S =
        OMPParallelDirective::CreateEmpty(Context,
                                          Record[ASTStmtReader::NumStmtFields],
                                          Empty);
      break;

    case STMT_OMP_SIMD_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPSimdDirective::CreateEmpty(Context, NumClauses,
                                        CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_FOR_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPForDirective::CreateEmpty(Context, NumClauses, CollapsedNum,
                                       Empty);
      break;
    }

    case STMT_OMP_FOR_SIMD_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPForSimdDirective::CreateEmpty(Context, NumClauses, CollapsedNum,
                                           Empty);
      break;
    }

    case STMT_OMP_SECTIONS_DIRECTIVE:
      S = OMPSectionsDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_SECTION_DIRECTIVE:
      S = OMPSectionDirective::CreateEmpty(Context, Empty);
      break;

    case STMT_OMP_SINGLE_DIRECTIVE:
      S = OMPSingleDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_MASTER_DIRECTIVE:
      S = OMPMasterDirective::CreateEmpty(Context, Empty);
      break;

    case STMT_OMP_CRITICAL_DIRECTIVE:
      S = OMPCriticalDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_PARALLEL_FOR_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPParallelForDirective::CreateEmpty(Context, NumClauses,
                                               CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_PARALLEL_FOR_SIMD_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPParallelForSimdDirective::CreateEmpty(Context, NumClauses,
                                                   CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_PARALLEL_SECTIONS_DIRECTIVE:
      S = OMPParallelSectionsDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TASK_DIRECTIVE:
      S = OMPTaskDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TASKYIELD_DIRECTIVE:
      S = OMPTaskyieldDirective::CreateEmpty(Context, Empty);
      break;

    case STMT_OMP_BARRIER_DIRECTIVE:
      S = OMPBarrierDirective::CreateEmpty(Context, Empty);
      break;

    case STMT_OMP_TASKWAIT_DIRECTIVE:
      S = OMPTaskwaitDirective::CreateEmpty(Context, Empty);
      break;

    case STMT_OMP_TASKGROUP_DIRECTIVE:
      S = OMPTaskgroupDirective::CreateEmpty(Context, Empty);
      break;

    case STMT_OMP_FLUSH_DIRECTIVE:
      S = OMPFlushDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_ORDERED_DIRECTIVE:
      S = OMPOrderedDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_ATOMIC_DIRECTIVE:
      S = OMPAtomicDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TARGET_DIRECTIVE:
      S = OMPTargetDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TARGET_DATA_DIRECTIVE:
      S = OMPTargetDataDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TARGET_ENTER_DATA_DIRECTIVE:
      S = OMPTargetEnterDataDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TARGET_EXIT_DATA_DIRECTIVE:
      S = OMPTargetExitDataDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TARGET_PARALLEL_DIRECTIVE:
      S = OMPTargetParallelDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TARGET_PARALLEL_FOR_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPTargetParallelForDirective::CreateEmpty(Context, NumClauses,
                                                     CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_TARGET_UPDATE_DIRECTIVE:
      S = OMPTargetUpdateDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TEAMS_DIRECTIVE:
      S = OMPTeamsDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_CANCELLATION_POINT_DIRECTIVE:
      S = OMPCancellationPointDirective::CreateEmpty(Context, Empty);
      break;

    case STMT_OMP_CANCEL_DIRECTIVE:
      S = OMPCancelDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;

    case STMT_OMP_TASKLOOP_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPTaskLoopDirective::CreateEmpty(Context, NumClauses, CollapsedNum,
                                            Empty);
      break;
    }

    case STMT_OMP_TASKLOOP_SIMD_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPTaskLoopSimdDirective::CreateEmpty(Context, NumClauses,
                                                CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_DISTRIBUTE_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPDistributeDirective::CreateEmpty(Context, NumClauses, CollapsedNum,
                                              Empty);
      break;
    }

    case STMT_OMP_DISTRIBUTE_PARALLEL_FOR_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPDistributeParallelForDirective::CreateEmpty(Context, NumClauses,
                                                         CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPDistributeParallelForSimdDirective::CreateEmpty(Context, NumClauses,
                                                             CollapsedNum,
                                                             Empty);
      break;
    }

    case STMT_OMP_DISTRIBUTE_SIMD_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPDistributeSimdDirective::CreateEmpty(Context, NumClauses,
                                                  CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_TARGET_PARALLEL_FOR_SIMD_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPTargetParallelForSimdDirective::CreateEmpty(Context, NumClauses,
                                                         CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_TARGET_SIMD_DIRECTIVE: {
      auto NumClauses = Record[ASTStmtReader::NumStmtFields];
      auto CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPTargetSimdDirective::CreateEmpty(Context, NumClauses, CollapsedNum,
                                              Empty);
      break;
    }

    case STMT_OMP_TEAMS_DISTRIBUTE_DIRECTIVE: {
      auto NumClauses = Record[ASTStmtReader::NumStmtFields];
      auto CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPTeamsDistributeDirective::CreateEmpty(Context, NumClauses,
                                                   CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_TEAMS_DISTRIBUTE_SIMD_DIRECTIVE: {
      unsigned NumClauses = Record[ASTStmtReader::NumStmtFields];
      unsigned CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPTeamsDistributeSimdDirective::CreateEmpty(Context, NumClauses,
                                                       CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_TEAMS_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE: {
      auto NumClauses = Record[ASTStmtReader::NumStmtFields];
      auto CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPTeamsDistributeParallelForSimdDirective::CreateEmpty(
          Context, NumClauses, CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_TEAMS_DISTRIBUTE_PARALLEL_FOR_DIRECTIVE: {
      auto NumClauses = Record[ASTStmtReader::NumStmtFields];
      auto CollapsedNum = Record[ASTStmtReader::NumStmtFields + 1];
      S = OMPTeamsDistributeParallelForDirective::CreateEmpty(
          Context, NumClauses, CollapsedNum, Empty);
      break;
    }

    case STMT_OMP_TARGET_TEAMS_DIRECTIVE: {
      S = OMPTargetTeamsDirective::CreateEmpty(
          Context, Record[ASTStmtReader::NumStmtFields], Empty);
      break;
    }
 
    case EXPR_CXX_OPERATOR_CALL:
      S = new (Context) CXXOperatorCallExpr(Context, Empty);
      break;

    case EXPR_CXX_MEMBER_CALL:
      S = new (Context) CXXMemberCallExpr(Context, Empty);
      break;

    case EXPR_CXX_CONSTRUCT:
      S = new (Context) CXXConstructExpr(Empty);
      break;

    case EXPR_CXX_INHERITED_CTOR_INIT:
      S = new (Context) CXXInheritedCtorInitExpr(Empty);
      break;

    case EXPR_CXX_TEMPORARY_OBJECT:
      S = new (Context) CXXTemporaryObjectExpr(Empty);
      break;

    case EXPR_CXX_STATIC_CAST:
      S = CXXStaticCastExpr::CreateEmpty(Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CXX_DYNAMIC_CAST:
      S = CXXDynamicCastExpr::CreateEmpty(Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CXX_REINTERPRET_CAST:
      S = CXXReinterpretCastExpr::CreateEmpty(Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CXX_CONST_CAST:
      S = CXXConstCastExpr::CreateEmpty(Context);
      break;

    case EXPR_CXX_FUNCTIONAL_CAST:
      S = CXXFunctionalCastExpr::CreateEmpty(Context,
                       /*PathSize*/ Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_USER_DEFINED_LITERAL:
      S = new (Context) UserDefinedLiteral(Context, Empty);
      break;

    case EXPR_CXX_STD_INITIALIZER_LIST:
      S = new (Context) CXXStdInitializerListExpr(Empty);
      break;

    case EXPR_CXX_BOOL_LITERAL:
      S = new (Context) CXXBoolLiteralExpr(Empty);
      break;

    case EXPR_CXX_NULL_PTR_LITERAL:
      S = new (Context) CXXNullPtrLiteralExpr(Empty);
      break;
    case EXPR_CXX_TYPEID_EXPR:
      S = new (Context) CXXTypeidExpr(Empty, true);
      break;
    case EXPR_CXX_TYPEID_TYPE:
      S = new (Context) CXXTypeidExpr(Empty, false);
      break;
    case EXPR_CXX_UUIDOF_EXPR:
      S = new (Context) CXXUuidofExpr(Empty, true);
      break;
    case EXPR_CXX_PROPERTY_REF_EXPR:
      S = new (Context) MSPropertyRefExpr(Empty);
      break;
    case EXPR_CXX_PROPERTY_SUBSCRIPT_EXPR:
      S = new (Context) MSPropertySubscriptExpr(Empty);
      break;
    case EXPR_CXX_UUIDOF_TYPE:
      S = new (Context) CXXUuidofExpr(Empty, false);
      break;
    case EXPR_CXX_THIS:
      S = new (Context) CXXThisExpr(Empty);
      break;
    case EXPR_CXX_THROW:
      S = new (Context) CXXThrowExpr(Empty);
      break;
    case EXPR_CXX_DEFAULT_ARG:
      S = new (Context) CXXDefaultArgExpr(Empty);
      break;
    case EXPR_CXX_DEFAULT_INIT:
      S = new (Context) CXXDefaultInitExpr(Empty);
      break;
    case EXPR_CXX_BIND_TEMPORARY:
      S = new (Context) CXXBindTemporaryExpr(Empty);
      break;

    case EXPR_CXX_SCALAR_VALUE_INIT:
      S = new (Context) CXXScalarValueInitExpr(Empty);
      break;
    case EXPR_CXX_NEW:
      S = new (Context) CXXNewExpr(Empty);
      break;
    case EXPR_CXX_DELETE:
      S = new (Context) CXXDeleteExpr(Empty);
      break;
    case EXPR_CXX_PSEUDO_DESTRUCTOR:
      S = new (Context) CXXPseudoDestructorExpr(Empty);
      break;

    case EXPR_EXPR_WITH_CLEANUPS:
      S = ExprWithCleanups::Create(Context, Empty,
                                   Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CXX_DEPENDENT_SCOPE_MEMBER:
      S = CXXDependentScopeMemberExpr::CreateEmpty(Context,
         /*HasTemplateKWAndArgsInfo=*/Record[ASTStmtReader::NumExprFields],
                  /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]
                                   ? Record[ASTStmtReader::NumExprFields + 1]
                                   : 0);
      break;

    case EXPR_CXX_DEPENDENT_SCOPE_DECL_REF:
      S = DependentScopeDeclRefExpr::CreateEmpty(Context,
         /*HasTemplateKWAndArgsInfo=*/Record[ASTStmtReader::NumExprFields],
                  /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]
                                   ? Record[ASTStmtReader::NumExprFields + 1]
                                   : 0);
      break;

    case EXPR_CXX_UNRESOLVED_CONSTRUCT:
      S = CXXUnresolvedConstructExpr::CreateEmpty(Context,
                              /*NumArgs=*/Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_CXX_UNRESOLVED_MEMBER:
      S = UnresolvedMemberExpr::CreateEmpty(Context,
         /*HasTemplateKWAndArgsInfo=*/Record[ASTStmtReader::NumExprFields],
                  /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]
                                   ? Record[ASTStmtReader::NumExprFields + 1]
                                   : 0);
      break;

    case EXPR_CXX_UNRESOLVED_LOOKUP:
      S = UnresolvedLookupExpr::CreateEmpty(Context,
         /*HasTemplateKWAndArgsInfo=*/Record[ASTStmtReader::NumExprFields],
                  /*NumTemplateArgs=*/Record[ASTStmtReader::NumExprFields]
                                   ? Record[ASTStmtReader::NumExprFields + 1]
                                   : 0);
      break;

    case EXPR_TYPE_TRAIT:
      S = TypeTraitExpr::CreateDeserialized(Context,
            Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_ARRAY_TYPE_TRAIT:
      S = new (Context) ArrayTypeTraitExpr(Empty);
      break;

    case EXPR_CXX_EXPRESSION_TRAIT:
      S = new (Context) ExpressionTraitExpr(Empty);
      break;

    case EXPR_CXX_NOEXCEPT:
      S = new (Context) CXXNoexceptExpr(Empty);
      break;

    case EXPR_PACK_EXPANSION:
      S = new (Context) PackExpansionExpr(Empty);
      break;

    case EXPR_SIZEOF_PACK:
      S = SizeOfPackExpr::CreateDeserialized(
              Context,
              /*NumPartialArgs=*/Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_SUBST_NON_TYPE_TEMPLATE_PARM:
      S = new (Context) SubstNonTypeTemplateParmExpr(Empty);
      break;

    case EXPR_SUBST_NON_TYPE_TEMPLATE_PARM_PACK:
      S = new (Context) SubstNonTypeTemplateParmPackExpr(Empty);
      break;

    case EXPR_FUNCTION_PARM_PACK:
      S = FunctionParmPackExpr::CreateEmpty(Context,
                                          Record[ASTStmtReader::NumExprFields]);
      break;

    case EXPR_MATERIALIZE_TEMPORARY:
      S = new (Context) MaterializeTemporaryExpr(Empty);
      break;

    case EXPR_CXX_FOLD:
      S = new (Context) CXXFoldExpr(Empty);
      break;

    case EXPR_OPAQUE_VALUE:
      S = new (Context) OpaqueValueExpr(Empty);
      break;

    case EXPR_CUDA_KERNEL_CALL:
      S = new (Context) CUDAKernelCallExpr(Context, Empty);
      break;

    case EXPR_ASTYPE:
      S = new (Context) AsTypeExpr(Empty);
      break;

    case EXPR_PSEUDO_OBJECT: {
      unsigned numSemanticExprs = Record[ASTStmtReader::NumExprFields];
      S = PseudoObjectExpr::Create(Context, Empty, numSemanticExprs);
      break;
    }

    case EXPR_ATOMIC:
      S = new (Context) AtomicExpr(Empty);
      break;

    case EXPR_LAMBDA: {
      unsigned NumCaptures = Record[ASTStmtReader::NumExprFields];
      S = LambdaExpr::CreateDeserialized(Context, NumCaptures);
      break;
    }
    }

    // We hit a STMT_STOP, so we're done with this expression.
    if (Finished)
      break;

    ++NumStatementsRead;

    if (S && !IsStmtReference) {
      Reader.Visit(S);
      StmtEntries[Cursor.GetCurrentBitNo()] = S;
    }


    assert(Idx == Record.size() && "Invalid deserialization of statement");
    StmtStack.push_back(S);
  }
Done:
  assert(StmtStack.size() > PrevNumStmts && "Read too many sub-stmts!");
  assert(StmtStack.size() == PrevNumStmts + 1 && "Extra expressions on stack!");
  return StmtStack.pop_back_val();
}
