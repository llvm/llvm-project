//===--- SemaStmtAsm.cpp - Semantic Analysis for Asm Statements -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for inline asm statements.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
using namespace clang;
using namespace sema;

/// CheckAsmLValue - GNU C has an extremely ugly extension whereby they silently
/// ignore "noop" casts in places where an lvalue is required by an inline asm.
/// We emulate this behavior when -fheinous-gnu-extensions is specified, but
/// provide a strong guidance to not use it.
///
/// This method checks to see if the argument is an acceptable l-value and
/// returns false if it is a case we can handle.
static bool CheckAsmLValue(const Expr *E, Sema &S) {
  // Type dependent expressions will be checked during instantiation.
  if (E->isTypeDependent())
    return false;

  if (E->isLValue())
    return false;  // Cool, this is an lvalue.

  // Okay, this is not an lvalue, but perhaps it is the result of a cast that we
  // are supposed to allow.
  const Expr *E2 = E->IgnoreParenNoopCasts(S.Context);
  if (E != E2 && E2->isLValue()) {
    if (!S.getLangOpts().HeinousExtensions)
      S.Diag(E2->getLocStart(), diag::err_invalid_asm_cast_lvalue)
        << E->getSourceRange();
    else
      S.Diag(E2->getLocStart(), diag::warn_invalid_asm_cast_lvalue)
        << E->getSourceRange();
    // Accept, even if we emitted an error diagnostic.
    return false;
  }

  // None of the above, just randomly invalid non-lvalue.
  return true;
}

/// isOperandMentioned - Return true if the specified operand # is mentioned
/// anywhere in the decomposed asm string.
static bool isOperandMentioned(unsigned OpNo,
                         ArrayRef<GCCAsmStmt::AsmStringPiece> AsmStrPieces) {
  for (unsigned p = 0, e = AsmStrPieces.size(); p != e; ++p) {
    const GCCAsmStmt::AsmStringPiece &Piece = AsmStrPieces[p];
    if (!Piece.isOperand()) continue;

    // If this is a reference to the input and if the input was the smaller
    // one, then we have to reject this asm.
    if (Piece.getOperandNo() == OpNo)
      return true;
  }
  return false;
}

StmtResult Sema::ActOnGCCAsmStmt(SourceLocation AsmLoc, bool IsSimple,
                                 bool IsVolatile, unsigned NumOutputs,
                                 unsigned NumInputs, IdentifierInfo **Names,
                                 MultiExprArg constraints, MultiExprArg exprs,
                                 Expr *asmString, MultiExprArg clobbers,
                                 SourceLocation RParenLoc) {
  unsigned NumClobbers = clobbers.size();
  StringLiteral **Constraints =
    reinterpret_cast<StringLiteral**>(constraints.data());
  Expr **Exprs = exprs.data();
  StringLiteral *AsmString = cast<StringLiteral>(asmString);
  StringLiteral **Clobbers = reinterpret_cast<StringLiteral**>(clobbers.data());

  SmallVector<TargetInfo::ConstraintInfo, 4> OutputConstraintInfos;

  // The parser verifies that there is a string literal here.
  if (!AsmString->isAscii())
    return StmtError(Diag(AsmString->getLocStart(),diag::err_asm_wide_character)
      << AsmString->getSourceRange());

  for (unsigned i = 0; i != NumOutputs; i++) {
    StringLiteral *Literal = Constraints[i];
    if (!Literal->isAscii())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    StringRef OutputName;
    if (Names[i])
      OutputName = Names[i]->getName();

    TargetInfo::ConstraintInfo Info(Literal->getString(), OutputName);
    if (!Context.getTargetInfo().validateOutputConstraint(Info))
      return StmtError(Diag(Literal->getLocStart(),
                            diag::err_asm_invalid_output_constraint)
                       << Info.getConstraintStr());

    // Check that the output exprs are valid lvalues.
    Expr *OutputExpr = Exprs[i];
    if (CheckAsmLValue(OutputExpr, *this)) {
      return StmtError(Diag(OutputExpr->getLocStart(),
                  diag::err_asm_invalid_lvalue_in_output)
        << OutputExpr->getSourceRange());
    }

    OutputConstraintInfos.push_back(Info);
  }

  SmallVector<TargetInfo::ConstraintInfo, 4> InputConstraintInfos;

  for (unsigned i = NumOutputs, e = NumOutputs + NumInputs; i != e; i++) {
    StringLiteral *Literal = Constraints[i];
    if (!Literal->isAscii())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    StringRef InputName;
    if (Names[i])
      InputName = Names[i]->getName();

    TargetInfo::ConstraintInfo Info(Literal->getString(), InputName);
    if (!Context.getTargetInfo().validateInputConstraint(OutputConstraintInfos.data(),
                                                NumOutputs, Info)) {
      return StmtError(Diag(Literal->getLocStart(),
                            diag::err_asm_invalid_input_constraint)
                       << Info.getConstraintStr());
    }

    Expr *InputExpr = Exprs[i];

    // Only allow void types for memory constraints.
    if (Info.allowsMemory() && !Info.allowsRegister()) {
      if (CheckAsmLValue(InputExpr, *this))
        return StmtError(Diag(InputExpr->getLocStart(),
                              diag::err_asm_invalid_lvalue_in_input)
                         << Info.getConstraintStr()
                         << InputExpr->getSourceRange());
    }

    if (Info.allowsRegister()) {
      if (InputExpr->getType()->isVoidType()) {
        return StmtError(Diag(InputExpr->getLocStart(),
                              diag::err_asm_invalid_type_in_input)
          << InputExpr->getType() << Info.getConstraintStr()
          << InputExpr->getSourceRange());
      }
    }

    ExprResult Result = DefaultFunctionArrayLvalueConversion(Exprs[i]);
    if (Result.isInvalid())
      return StmtError();

    Exprs[i] = Result.take();
    InputConstraintInfos.push_back(Info);
  }

  // Check that the clobbers are valid.
  for (unsigned i = 0; i != NumClobbers; i++) {
    StringLiteral *Literal = Clobbers[i];
    if (!Literal->isAscii())
      return StmtError(Diag(Literal->getLocStart(),diag::err_asm_wide_character)
        << Literal->getSourceRange());

    StringRef Clobber = Literal->getString();

    if (!Context.getTargetInfo().isValidClobber(Clobber))
      return StmtError(Diag(Literal->getLocStart(),
                  diag::err_asm_unknown_register_name) << Clobber);
  }

  GCCAsmStmt *NS =
    new (Context) GCCAsmStmt(Context, AsmLoc, IsSimple, IsVolatile, NumOutputs,
                             NumInputs, Names, Constraints, Exprs, AsmString,
                             NumClobbers, Clobbers, RParenLoc);
  // Validate the asm string, ensuring it makes sense given the operands we
  // have.
  SmallVector<GCCAsmStmt::AsmStringPiece, 8> Pieces;
  unsigned DiagOffs;
  if (unsigned DiagID = NS->AnalyzeAsmString(Pieces, Context, DiagOffs)) {
    Diag(getLocationOfStringLiteralByte(AsmString, DiagOffs), DiagID)
           << AsmString->getSourceRange();
    return StmtError();
  }

  // Validate constraints and modifiers.
  for (unsigned i = 0, e = Pieces.size(); i != e; ++i) {
    GCCAsmStmt::AsmStringPiece &Piece = Pieces[i];
    if (!Piece.isOperand()) continue;

    // Look for the correct constraint index.
    unsigned Idx = 0;
    unsigned ConstraintIdx = 0;
    for (unsigned i = 0, e = NS->getNumOutputs(); i != e; ++i, ++ConstraintIdx) {
      TargetInfo::ConstraintInfo &Info = OutputConstraintInfos[i];
      if (Idx == Piece.getOperandNo())
        break;
      ++Idx;

      if (Info.isReadWrite()) {
        if (Idx == Piece.getOperandNo())
          break;
        ++Idx;
      }
    }

    for (unsigned i = 0, e = NS->getNumInputs(); i != e; ++i, ++ConstraintIdx) {
      TargetInfo::ConstraintInfo &Info = InputConstraintInfos[i];
      if (Idx == Piece.getOperandNo())
        break;
      ++Idx;

      if (Info.isReadWrite()) {
        if (Idx == Piece.getOperandNo())
          break;
        ++Idx;
      }
    }

    // Now that we have the right indexes go ahead and check.
    StringLiteral *Literal = Constraints[ConstraintIdx];
    const Type *Ty = Exprs[ConstraintIdx]->getType().getTypePtr();
    if (Ty->isDependentType() || Ty->isIncompleteType())
      continue;

    unsigned Size = Context.getTypeSize(Ty);
    if (!Context.getTargetInfo()
          .validateConstraintModifier(Literal->getString(), Piece.getModifier(),
                                      Size))
      Diag(Exprs[ConstraintIdx]->getLocStart(),
           diag::warn_asm_mismatched_size_modifier);
  }

  // Validate tied input operands for type mismatches.
  for (unsigned i = 0, e = InputConstraintInfos.size(); i != e; ++i) {
    TargetInfo::ConstraintInfo &Info = InputConstraintInfos[i];

    // If this is a tied constraint, verify that the output and input have
    // either exactly the same type, or that they are int/ptr operands with the
    // same size (int/long, int*/long, are ok etc).
    if (!Info.hasTiedOperand()) continue;

    unsigned TiedTo = Info.getTiedOperand();
    unsigned InputOpNo = i+NumOutputs;
    Expr *OutputExpr = Exprs[TiedTo];
    Expr *InputExpr = Exprs[InputOpNo];

    if (OutputExpr->isTypeDependent() || InputExpr->isTypeDependent())
      continue;

    QualType InTy = InputExpr->getType();
    QualType OutTy = OutputExpr->getType();
    if (Context.hasSameType(InTy, OutTy))
      continue;  // All types can be tied to themselves.

    // Decide if the input and output are in the same domain (integer/ptr or
    // floating point.
    enum AsmDomain {
      AD_Int, AD_FP, AD_Other
    } InputDomain, OutputDomain;

    if (InTy->isIntegerType() || InTy->isPointerType())
      InputDomain = AD_Int;
    else if (InTy->isRealFloatingType())
      InputDomain = AD_FP;
    else
      InputDomain = AD_Other;

    if (OutTy->isIntegerType() || OutTy->isPointerType())
      OutputDomain = AD_Int;
    else if (OutTy->isRealFloatingType())
      OutputDomain = AD_FP;
    else
      OutputDomain = AD_Other;

    // They are ok if they are the same size and in the same domain.  This
    // allows tying things like:
    //   void* to int*
    //   void* to int            if they are the same size.
    //   double to long double   if they are the same size.
    //
    uint64_t OutSize = Context.getTypeSize(OutTy);
    uint64_t InSize = Context.getTypeSize(InTy);
    if (OutSize == InSize && InputDomain == OutputDomain &&
        InputDomain != AD_Other)
      continue;

    // If the smaller input/output operand is not mentioned in the asm string,
    // then we can promote the smaller one to a larger input and the asm string
    // won't notice.
    bool SmallerValueMentioned = false;

    // If this is a reference to the input and if the input was the smaller
    // one, then we have to reject this asm.
    if (isOperandMentioned(InputOpNo, Pieces)) {
      // This is a use in the asm string of the smaller operand.  Since we
      // codegen this by promoting to a wider value, the asm will get printed
      // "wrong".
      SmallerValueMentioned |= InSize < OutSize;
    }
    if (isOperandMentioned(TiedTo, Pieces)) {
      // If this is a reference to the output, and if the output is the larger
      // value, then it's ok because we'll promote the input to the larger type.
      SmallerValueMentioned |= OutSize < InSize;
    }

    // If the smaller value wasn't mentioned in the asm string, and if the
    // output was a register, just extend the shorter one to the size of the
    // larger one.
    if (!SmallerValueMentioned && InputDomain != AD_Other &&
        OutputConstraintInfos[TiedTo].allowsRegister())
      continue;

    // Either both of the operands were mentioned or the smaller one was
    // mentioned.  One more special case that we'll allow: if the tied input is
    // integer, unmentioned, and is a constant, then we'll allow truncating it
    // down to the size of the destination.
    if (InputDomain == AD_Int && OutputDomain == AD_Int &&
        !isOperandMentioned(InputOpNo, Pieces) &&
        InputExpr->isEvaluatable(Context)) {
      CastKind castKind =
        (OutTy->isBooleanType() ? CK_IntegralToBoolean : CK_IntegralCast);
      InputExpr = ImpCastExprToType(InputExpr, OutTy, castKind).take();
      Exprs[InputOpNo] = InputExpr;
      NS->setInputExpr(i, InputExpr);
      continue;
    }

    Diag(InputExpr->getLocStart(),
         diag::err_asm_tying_incompatible_types)
      << InTy << OutTy << OutputExpr->getSourceRange()
      << InputExpr->getSourceRange();
    return StmtError();
  }

  return Owned(NS);
}

// getSpelling - Get the spelling of the AsmTok token.
static StringRef getSpelling(Sema &SemaRef, Token AsmTok) {
  StringRef Asm;
  SmallString<512> TokenBuf;
  TokenBuf.resize(512);
  bool StringInvalid = false;
  Asm = SemaRef.PP.getSpelling(AsmTok, TokenBuf, &StringInvalid);
  assert (!StringInvalid && "Expected valid string!");
  return Asm;
}

// Build the inline assembly string.  Returns true on error.
static bool buildMSAsmString(Sema &SemaRef,
                             SourceLocation AsmLoc,
                             ArrayRef<Token> AsmToks,
                             llvm::SmallVectorImpl<unsigned> &TokOffsets,
                             std::string &AsmString) {
  assert (!AsmToks.empty() && "Didn't expect an empty AsmToks!");

  SmallString<512> Asm;
  for (unsigned i = 0, e = AsmToks.size(); i < e; ++i) {
    bool isNewAsm = ((i == 0) ||
                     AsmToks[i].isAtStartOfLine() ||
                     AsmToks[i].is(tok::kw_asm));
    if (isNewAsm) {
      if (i != 0)
        Asm += "\n\t";

      if (AsmToks[i].is(tok::kw_asm)) {
        i++; // Skip __asm
        if (i == e) {
          SemaRef.Diag(AsmLoc, diag::err_asm_empty);
          return true;
        }

      }
    }

    if (i && AsmToks[i].hasLeadingSpace() && !isNewAsm)
      Asm += ' ';

    StringRef Spelling = getSpelling(SemaRef, AsmToks[i]);
    Asm += Spelling;
    TokOffsets.push_back(Asm.size());
  }
  AsmString = Asm.str();
  return false;
}

namespace {

class MCAsmParserSemaCallbackImpl : public llvm::MCAsmParserSemaCallback {
  Sema &SemaRef;
  SourceLocation AsmLoc;
  ArrayRef<Token> AsmToks;
  ArrayRef<unsigned> TokOffsets;

public:
  MCAsmParserSemaCallbackImpl(Sema &Ref, SourceLocation Loc,
                              ArrayRef<Token> Toks,
                              ArrayRef<unsigned> Offsets)
    : SemaRef(Ref), AsmLoc(Loc), AsmToks(Toks), TokOffsets(Offsets) { }
  ~MCAsmParserSemaCallbackImpl() {}

  void *LookupInlineAsmIdentifier(StringRef Name, void *SrcLoc, unsigned &Size){
    SourceLocation Loc = SourceLocation::getFromPtrEncoding(SrcLoc);
    NamedDecl *OpDecl = SemaRef.LookupInlineAsmIdentifier(Name, Loc, Size);
    return static_cast<void *>(OpDecl);
  }

  bool LookupInlineAsmField(StringRef Base, StringRef Member,
                            unsigned &Offset) {
    return SemaRef.LookupInlineAsmField(Base, Member, Offset, AsmLoc);
  }

  static void MSAsmDiagHandlerCallback(const llvm::SMDiagnostic &D,
                                       void *Context) {
    ((MCAsmParserSemaCallbackImpl*)Context)->MSAsmDiagHandler(D);
  }
  void MSAsmDiagHandler(const llvm::SMDiagnostic &D) {
    // Compute an offset into the inline asm buffer.
    // FIXME: This isn't right if .macro is involved (but hopefully, no
    // real-world code does that).
    const llvm::SourceMgr &LSM = *D.getSourceMgr();
    const llvm::MemoryBuffer *LBuf =
    LSM.getMemoryBuffer(LSM.FindBufferContainingLoc(D.getLoc()));
    unsigned Offset = D.getLoc().getPointer()  - LBuf->getBufferStart();

    // Figure out which token that offset points into.
    const unsigned *OffsetPtr =
        std::lower_bound(TokOffsets.begin(), TokOffsets.end(), Offset);
    unsigned TokIndex = OffsetPtr - TokOffsets.begin();

    // If we come up with an answer which seems sane, use it; otherwise,
    // just point at the __asm keyword.
    // FIXME: Assert the answer is sane once we handle .macro correctly.
    SourceLocation Loc = AsmLoc;
    if (TokIndex < AsmToks.size()) {
      const Token *Tok = &AsmToks[TokIndex];
      Loc = Tok->getLocation();
      Loc = Loc.getLocWithOffset(Offset - (*OffsetPtr - Tok->getLength()));
    }
    SemaRef.Diag(Loc, diag::err_inline_ms_asm_parsing) << D.getMessage();
  }
};

}

NamedDecl *Sema::LookupInlineAsmIdentifier(StringRef Name, SourceLocation Loc,
                                           unsigned &Size) {
  Size = 0;
  LookupResult Result(*this, &Context.Idents.get(Name), Loc,
                      Sema::LookupOrdinaryName);

  if (!LookupName(Result, getCurScope())) {
    // If we don't find anything, return null; the AsmParser will assume
    // it is a label of some sort.
    return 0;
  }

  if (!Result.isSingleResult()) {
    // FIXME: Diagnose result.
    return 0;
  }

  NamedDecl *ND = Result.getFoundDecl();
  if (isa<VarDecl>(ND) || isa<FunctionDecl>(ND)) {
    if (VarDecl *Var = dyn_cast<VarDecl>(ND))
      Size = Context.getTypeInfo(Var->getType()).first;

    return ND;
  }

  // FIXME: Handle other kinds of results? (FieldDecl, etc.)
  // FIXME: Diagnose if we find something we can't handle, like a typedef.
  return 0;
}

bool Sema::LookupInlineAsmField(StringRef Base, StringRef Member,
                                unsigned &Offset, SourceLocation AsmLoc) {
  Offset = 0;
  LookupResult BaseResult(*this, &Context.Idents.get(Base), SourceLocation(),
                          LookupOrdinaryName);

  if (!LookupName(BaseResult, getCurScope()))
    return true;

  if (!BaseResult.isSingleResult())
    return true;

  NamedDecl *FoundDecl = BaseResult.getFoundDecl();
  const RecordType *RT = 0;
  if (VarDecl *VD = dyn_cast<VarDecl>(FoundDecl)) {
    RT = VD->getType()->getAs<RecordType>();
  } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(FoundDecl)) {
    RT = TD->getUnderlyingType()->getAs<RecordType>();
  }
  if (!RT)
    return true;

  if (RequireCompleteType(AsmLoc, QualType(RT, 0), 0))
    return true;

  LookupResult FieldResult(*this, &Context.Idents.get(Member), SourceLocation(),
                           LookupMemberName);

  if (!LookupQualifiedName(FieldResult, RT->getDecl()))
    return true;

  // FIXME: Handle IndirectFieldDecl?
  FieldDecl *FD = dyn_cast<FieldDecl>(FieldResult.getFoundDecl());
  if (!FD)
    return true;

  const ASTRecordLayout &RL = Context.getASTRecordLayout(RT->getDecl());
  unsigned i = FD->getFieldIndex();
  CharUnits Result = Context.toCharUnitsFromBits(RL.getFieldOffset(i));
  Offset = (unsigned)Result.getQuantity();

  return false;
}

StmtResult Sema::ActOnMSAsmStmt(SourceLocation AsmLoc, SourceLocation LBraceLoc,
                                ArrayRef<Token> AsmToks,SourceLocation EndLoc) {
  SmallVector<IdentifierInfo*, 4> Names;
  SmallVector<StringRef, 4> ConstraintRefs;
  SmallVector<Expr*, 4> Exprs;
  SmallVector<StringRef, 4> ClobberRefs;

  // Empty asm statements don't need to instantiate the AsmParser, etc.
  if (AsmToks.empty()) {
    StringRef EmptyAsmStr;
    MSAsmStmt *NS =
      new (Context) MSAsmStmt(Context, AsmLoc, LBraceLoc, /*IsSimple*/ true,
                              /*IsVolatile*/ true, AsmToks, /*NumOutputs*/ 0,
                              /*NumInputs*/ 0, Names, ConstraintRefs, Exprs,
                              EmptyAsmStr, ClobberRefs, EndLoc);
    return Owned(NS);
  }

  std::string AsmString;
  llvm::SmallVector<unsigned, 8> TokOffsets;
  if (buildMSAsmString(*this, AsmLoc, AsmToks, TokOffsets, AsmString))
    return StmtError();

  // Get the target specific parser.
  std::string Error;
  const std::string &TT = Context.getTargetInfo().getTriple().getTriple();
  const llvm::Target *TheTarget(llvm::TargetRegistry::lookupTarget(TT, Error));

  OwningPtr<llvm::MCAsmInfo> MAI(TheTarget->createMCAsmInfo(TT));
  OwningPtr<llvm::MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
  OwningPtr<llvm::MCObjectFileInfo> MOFI(new llvm::MCObjectFileInfo());
  OwningPtr<llvm::MCSubtargetInfo>
    STI(TheTarget->createMCSubtargetInfo(TT, "", ""));

  llvm::SourceMgr SrcMgr;
  llvm::MCContext Ctx(*MAI, *MRI, MOFI.get(), &SrcMgr);
  llvm::MemoryBuffer *Buffer =
    llvm::MemoryBuffer::getMemBuffer(AsmString, "<inline asm>");

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(Buffer, llvm::SMLoc());

  OwningPtr<llvm::MCStreamer> Str(createNullStreamer(Ctx));
  OwningPtr<llvm::MCAsmParser>
    Parser(createMCAsmParser(SrcMgr, Ctx, *Str.get(), *MAI));
  OwningPtr<llvm::MCTargetAsmParser>
    TargetParser(TheTarget->createMCAsmParser(*STI, *Parser));

  // Get the instruction descriptor.
  const llvm::MCInstrInfo *MII = TheTarget->createMCInstrInfo(); 
  llvm::MCInstPrinter *IP =
    TheTarget->createMCInstPrinter(1, *MAI, *MII, *MRI, *STI);

  // Change to the Intel dialect.
  Parser->setAssemblerDialect(1);
  Parser->setTargetParser(*TargetParser.get());
  Parser->setParsingInlineAsm(true);
  TargetParser->setParsingInlineAsm(true);

  MCAsmParserSemaCallbackImpl MCAPSI(*this, AsmLoc, AsmToks, TokOffsets);
  TargetParser->setSemaCallback(&MCAPSI);
  SrcMgr.setDiagHandler(MCAsmParserSemaCallbackImpl::MSAsmDiagHandlerCallback,
                        &MCAPSI);

  unsigned NumOutputs;
  unsigned NumInputs;
  std::string AsmStringIR;
  SmallVector<std::pair<void *, bool>, 4> OpDecls;
  SmallVector<std::string, 4> Constraints;
  SmallVector<std::string, 4> Clobbers;
  if (Parser->ParseMSInlineAsm(AsmLoc.getPtrEncoding(), AsmStringIR,
                               NumOutputs, NumInputs, OpDecls, Constraints,
                               Clobbers, MII, IP, MCAPSI))
    return StmtError();

  // Build the vector of clobber StringRefs.
  unsigned NumClobbers = Clobbers.size();
  ClobberRefs.resize(NumClobbers);
  for (unsigned i = 0; i != NumClobbers; ++i)
    ClobberRefs[i] = StringRef(Clobbers[i]);

  // Recast the void pointers and build the vector of constraint StringRefs.
  unsigned NumExprs = NumOutputs + NumInputs;
  Names.resize(NumExprs);
  ConstraintRefs.resize(NumExprs);
  Exprs.resize(NumExprs);
  for (unsigned i = 0, e = NumExprs; i != e; ++i) {
    NamedDecl *OpDecl = static_cast<NamedDecl *>(OpDecls[i].first);
    if (!OpDecl)
      return StmtError();

    DeclarationNameInfo NameInfo(OpDecl->getDeclName(), AsmLoc);
    ExprResult OpExpr = BuildDeclarationNameExpr(CXXScopeSpec(), NameInfo,
                                                 OpDecl);
    if (OpExpr.isInvalid())
      return StmtError();

    // Need offset of variable.
    if (OpDecls[i].second)
      OpExpr = BuildUnaryOp(getCurScope(), AsmLoc, clang::UO_AddrOf,
                            OpExpr.take());

    Names[i] = OpDecl->getIdentifier();
    ConstraintRefs[i] = StringRef(Constraints[i]);
    Exprs[i] = OpExpr.take();
  }

  bool IsSimple = NumExprs > 0;
  MSAsmStmt *NS =
    new (Context) MSAsmStmt(Context, AsmLoc, LBraceLoc, IsSimple,
                            /*IsVolatile*/ true, AsmToks, NumOutputs, NumInputs,
                            Names, ConstraintRefs, Exprs, AsmStringIR,
                            ClobberRefs, EndLoc);
  return Owned(NS);
}
