//===--- Stmt.cpp - Statement AST Node Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt class and statement subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Stmt.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/Type.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

static struct StmtClassNameTable {
  const char *Name;
  unsigned Counter;
  unsigned Size;
} StmtClassInfo[Stmt::lastStmtConstant+1];

static StmtClassNameTable &getStmtInfoTableEntry(Stmt::StmtClass E) {
  static bool Initialized = false;
  if (Initialized)
    return StmtClassInfo[E];

  // Intialize the table on the first use.
  Initialized = true;
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT) \
  StmtClassInfo[(unsigned)Stmt::CLASS##Class].Name = #CLASS;    \
  StmtClassInfo[(unsigned)Stmt::CLASS##Class].Size = sizeof(CLASS);
#include "clang/AST/StmtNodes.inc"

  return StmtClassInfo[E];
}

const char *Stmt::getStmtClassName() const {
  return getStmtInfoTableEntry((StmtClass) StmtBits.sClass).Name;
}

void Stmt::PrintStats() {
  // Ensure the table is primed.
  getStmtInfoTableEntry(Stmt::NullStmtClass);

  unsigned sum = 0;
  llvm::errs() << "\n*** Stmt/Expr Stats:\n";
  for (int i = 0; i != Stmt::lastStmtConstant+1; i++) {
    if (StmtClassInfo[i].Name == 0) continue;
    sum += StmtClassInfo[i].Counter;
  }
  llvm::errs() << "  " << sum << " stmts/exprs total.\n";
  sum = 0;
  for (int i = 0; i != Stmt::lastStmtConstant+1; i++) {
    if (StmtClassInfo[i].Name == 0) continue;
    if (StmtClassInfo[i].Counter == 0) continue;
    llvm::errs() << "    " << StmtClassInfo[i].Counter << " "
                 << StmtClassInfo[i].Name << ", " << StmtClassInfo[i].Size
                 << " each (" << StmtClassInfo[i].Counter*StmtClassInfo[i].Size
                 << " bytes)\n";
    sum += StmtClassInfo[i].Counter*StmtClassInfo[i].Size;
  }

  llvm::errs() << "Total bytes = " << sum << "\n";
}

void Stmt::addStmtClass(StmtClass s) {
  ++getStmtInfoTableEntry(s).Counter;
}

bool Stmt::StatisticsEnabled = false;
void Stmt::EnableStatistics() {
  StatisticsEnabled = true;
}

Stmt *Stmt::IgnoreImplicit() {
  Stmt *s = this;

  if (ExprWithCleanups *ewc = dyn_cast<ExprWithCleanups>(s))
    s = ewc->getSubExpr();

  while (ImplicitCastExpr *ice = dyn_cast<ImplicitCastExpr>(s))
    s = ice->getSubExpr();

  return s;
}

/// \brief Strip off all label-like statements.
///
/// This will strip off label statements, case statements, attributed
/// statements and default statements recursively.
const Stmt *Stmt::stripLabelLikeStatements() const {
  const Stmt *S = this;
  while (true) {
    if (const LabelStmt *LS = dyn_cast<LabelStmt>(S))
      S = LS->getSubStmt();
    else if (const SwitchCase *SC = dyn_cast<SwitchCase>(S))
      S = SC->getSubStmt();
    else if (const AttributedStmt *AS = dyn_cast<AttributedStmt>(S))
      S = AS->getSubStmt();
    else
      return S;
  }
}

namespace {
  struct good {};
  struct bad {};

  // These silly little functions have to be static inline to suppress
  // unused warnings, and they have to be defined to suppress other
  // warnings.
  static inline good is_good(good) { return good(); }

  typedef Stmt::child_range children_t();
  template <class T> good implements_children(children_t T::*) {
    return good();
  }
  static inline bad implements_children(children_t Stmt::*) {
    return bad();
  }

  typedef SourceRange getSourceRange_t() const;
  template <class T> good implements_getSourceRange(getSourceRange_t T::*) {
    return good();
  }
  static inline bad implements_getSourceRange(getSourceRange_t Stmt::*) {
    return bad();
  }

#define ASSERT_IMPLEMENTS_children(type) \
  (void) sizeof(is_good(implements_children(&type::children)))
#define ASSERT_IMPLEMENTS_getSourceRange(type) \
  (void) sizeof(is_good(implements_getSourceRange(&type::getSourceRange)))
}

/// Check whether the various Stmt classes implement their member
/// functions.
static inline void check_implementations() {
#define ABSTRACT_STMT(type)
#define STMT(type, base) \
  ASSERT_IMPLEMENTS_children(type); \
  ASSERT_IMPLEMENTS_getSourceRange(type);
#include "clang/AST/StmtNodes.inc"
}

Stmt::child_range Stmt::children() {
  switch (getStmtClass()) {
  case Stmt::NoStmtClass: llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base) \
  case Stmt::type##Class: \
    return static_cast<type*>(this)->children();
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind!");
}

SourceRange Stmt::getSourceRange() const {
  switch (getStmtClass()) {
  case Stmt::NoStmtClass: llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base) \
  case Stmt::type##Class: \
    return static_cast<const type*>(this)->getSourceRange();
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind!");
}

// Amusing macro metaprogramming hack: check whether a class provides
// a more specific implementation of getLocStart() and getLocEnd().
//
// See also Expr.cpp:getExprLoc().
namespace {
  /// This implementation is used when a class provides a custom
  /// implementation of getLocStart.
  template <class S, class T>
  SourceLocation getLocStartImpl(const Stmt *stmt,
                                 SourceLocation (T::*v)() const) {
    return static_cast<const S*>(stmt)->getLocStart();
  }

  /// This implementation is used when a class doesn't provide a custom
  /// implementation of getLocStart.  Overload resolution should pick it over
  /// the implementation above because it's more specialized according to
  /// function template partial ordering.
  template <class S>
  SourceLocation getLocStartImpl(const Stmt *stmt,
                                SourceLocation (Stmt::*v)() const) {
    return static_cast<const S*>(stmt)->getSourceRange().getBegin();
  }

  /// This implementation is used when a class provides a custom
  /// implementation of getLocEnd.
  template <class S, class T>
  SourceLocation getLocEndImpl(const Stmt *stmt,
                               SourceLocation (T::*v)() const) {
    return static_cast<const S*>(stmt)->getLocEnd();
  }

  /// This implementation is used when a class doesn't provide a custom
  /// implementation of getLocEnd.  Overload resolution should pick it over
  /// the implementation above because it's more specialized according to
  /// function template partial ordering.
  template <class S>
  SourceLocation getLocEndImpl(const Stmt *stmt,
                               SourceLocation (Stmt::*v)() const) {
    return static_cast<const S*>(stmt)->getSourceRange().getEnd();
  }
}

SourceLocation Stmt::getLocStart() const {
  switch (getStmtClass()) {
  case Stmt::NoStmtClass: llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base) \
  case Stmt::type##Class: \
    return getLocStartImpl<type>(this, &type::getLocStart);
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind");
}

SourceLocation Stmt::getLocEnd() const {
  switch (getStmtClass()) {
  case Stmt::NoStmtClass: llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base) \
  case Stmt::type##Class: \
    return getLocEndImpl<type>(this, &type::getLocEnd);
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind");
}

CompoundStmt::CompoundStmt(ASTContext &C, Stmt **StmtStart, unsigned NumStmts,
                           SourceLocation LB, SourceLocation RB)
  : Stmt(CompoundStmtClass), LBracLoc(LB), RBracLoc(RB) {
  CompoundStmtBits.NumStmts = NumStmts;
  assert(CompoundStmtBits.NumStmts == NumStmts &&
         "NumStmts doesn't fit in bits of CompoundStmtBits.NumStmts!");

  if (NumStmts == 0) {
    Body = 0;
    return;
  }

  Body = new (C) Stmt*[NumStmts];
  memcpy(Body, StmtStart, NumStmts * sizeof(*Body));
}

void CompoundStmt::setStmts(ASTContext &C, Stmt **Stmts, unsigned NumStmts) {
  if (this->Body)
    C.Deallocate(Body);
  this->CompoundStmtBits.NumStmts = NumStmts;

  Body = new (C) Stmt*[NumStmts];
  memcpy(Body, Stmts, sizeof(Stmt *) * NumStmts);
}

const char *LabelStmt::getName() const {
  return getDecl()->getIdentifier()->getNameStart();
}

AttributedStmt *AttributedStmt::Create(ASTContext &C, SourceLocation Loc,
                                       ArrayRef<const Attr*> Attrs,
                                       Stmt *SubStmt) {
  void *Mem = C.Allocate(sizeof(AttributedStmt) +
                         sizeof(Attr*) * (Attrs.size() - 1),
                         llvm::alignOf<AttributedStmt>());
  return new (Mem) AttributedStmt(Loc, Attrs, SubStmt);
}

AttributedStmt *AttributedStmt::CreateEmpty(ASTContext &C, unsigned NumAttrs) {
  assert(NumAttrs > 0 && "NumAttrs should be greater than zero");
  void *Mem = C.Allocate(sizeof(AttributedStmt) +
                         sizeof(Attr*) * (NumAttrs - 1),
                         llvm::alignOf<AttributedStmt>());
  return new (Mem) AttributedStmt(EmptyShell(), NumAttrs);
}

// This is defined here to avoid polluting Stmt.h with importing Expr.h
SourceRange ReturnStmt::getSourceRange() const {
  if (RetExpr)
    return SourceRange(RetLoc, RetExpr->getLocEnd());
  else
    return SourceRange(RetLoc);
}

bool Stmt::hasImplicitControlFlow() const {
  switch (StmtBits.sClass) {
    default:
      return false;

    case CallExprClass:
    case ConditionalOperatorClass:
    case ChooseExprClass:
    case StmtExprClass:
    case DeclStmtClass:
      return true;

    case Stmt::BinaryOperatorClass: {
      const BinaryOperator* B = cast<BinaryOperator>(this);
      if (B->isLogicalOp() || B->getOpcode() == BO_Comma)
        return true;
      else
        return false;
    }
  }
}

std::string AsmStmt::generateAsmString(ASTContext &C) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->generateAsmString(C);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->generateAsmString(C);
  llvm_unreachable("unknown asm statement kind!");
}

StringRef AsmStmt::getOutputConstraint(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getOutputConstraint(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getOutputConstraint(i);
  llvm_unreachable("unknown asm statement kind!");
}

const Expr *AsmStmt::getOutputExpr(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getOutputExpr(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getOutputExpr(i);
  llvm_unreachable("unknown asm statement kind!");
}

StringRef AsmStmt::getInputConstraint(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getInputConstraint(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getInputConstraint(i);
  llvm_unreachable("unknown asm statement kind!");
}

const Expr *AsmStmt::getInputExpr(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getInputExpr(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getInputExpr(i);
  llvm_unreachable("unknown asm statement kind!");
}

StringRef AsmStmt::getClobber(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getClobber(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getClobber(i);
  llvm_unreachable("unknown asm statement kind!");
}

/// getNumPlusOperands - Return the number of output operands that have a "+"
/// constraint.
unsigned AsmStmt::getNumPlusOperands() const {
  unsigned Res = 0;
  for (unsigned i = 0, e = getNumOutputs(); i != e; ++i)
    if (isOutputPlusConstraint(i))
      ++Res;
  return Res;
}

StringRef GCCAsmStmt::getClobber(unsigned i) const {
  return getClobberStringLiteral(i)->getString();
}

Expr *GCCAsmStmt::getOutputExpr(unsigned i) {
  return cast<Expr>(Exprs[i]);
}

/// getOutputConstraint - Return the constraint string for the specified
/// output operand.  All output constraints are known to be non-empty (either
/// '=' or '+').
StringRef GCCAsmStmt::getOutputConstraint(unsigned i) const {
  return getOutputConstraintLiteral(i)->getString();
}

Expr *GCCAsmStmt::getInputExpr(unsigned i) {
  return cast<Expr>(Exprs[i + NumOutputs]);
}
void GCCAsmStmt::setInputExpr(unsigned i, Expr *E) {
  Exprs[i + NumOutputs] = E;
}

/// getInputConstraint - Return the specified input constraint.  Unlike output
/// constraints, these can be empty.
StringRef GCCAsmStmt::getInputConstraint(unsigned i) const {
  return getInputConstraintLiteral(i)->getString();
}

void GCCAsmStmt::setOutputsAndInputsAndClobbers(ASTContext &C,
                                             IdentifierInfo **Names,
                                             StringLiteral **Constraints,
                                             Stmt **Exprs,
                                             unsigned NumOutputs,
                                             unsigned NumInputs,
                                             StringLiteral **Clobbers,
                                             unsigned NumClobbers) {
  this->NumOutputs = NumOutputs;
  this->NumInputs = NumInputs;
  this->NumClobbers = NumClobbers;

  unsigned NumExprs = NumOutputs + NumInputs;

  C.Deallocate(this->Names);
  this->Names = new (C) IdentifierInfo*[NumExprs];
  std::copy(Names, Names + NumExprs, this->Names);

  C.Deallocate(this->Exprs);
  this->Exprs = new (C) Stmt*[NumExprs];
  std::copy(Exprs, Exprs + NumExprs, this->Exprs);

  C.Deallocate(this->Constraints);
  this->Constraints = new (C) StringLiteral*[NumExprs];
  std::copy(Constraints, Constraints + NumExprs, this->Constraints);

  C.Deallocate(this->Clobbers);
  this->Clobbers = new (C) StringLiteral*[NumClobbers];
  std::copy(Clobbers, Clobbers + NumClobbers, this->Clobbers);
}

/// getNamedOperand - Given a symbolic operand reference like %[foo],
/// translate this into a numeric value needed to reference the same operand.
/// This returns -1 if the operand name is invalid.
int GCCAsmStmt::getNamedOperand(StringRef SymbolicName) const {
  unsigned NumPlusOperands = 0;

  // Check if this is an output operand.
  for (unsigned i = 0, e = getNumOutputs(); i != e; ++i) {
    if (getOutputName(i) == SymbolicName)
      return i;
  }

  for (unsigned i = 0, e = getNumInputs(); i != e; ++i)
    if (getInputName(i) == SymbolicName)
      return getNumOutputs() + NumPlusOperands + i;

  // Not found.
  return -1;
}

/// AnalyzeAsmString - Analyze the asm string of the current asm, decomposing
/// it into pieces.  If the asm string is erroneous, emit errors and return
/// true, otherwise return false.
unsigned GCCAsmStmt::AnalyzeAsmString(SmallVectorImpl<AsmStringPiece>&Pieces,
                                   ASTContext &C, unsigned &DiagOffs) const {
  StringRef Str = getAsmString()->getString();
  const char *StrStart = Str.begin();
  const char *StrEnd = Str.end();
  const char *CurPtr = StrStart;

  // "Simple" inline asms have no constraints or operands, just convert the asm
  // string to escape $'s.
  if (isSimple()) {
    std::string Result;
    for (; CurPtr != StrEnd; ++CurPtr) {
      switch (*CurPtr) {
      case '$':
        Result += "$$";
        break;
      default:
        Result += *CurPtr;
        break;
      }
    }
    Pieces.push_back(AsmStringPiece(Result));
    return 0;
  }

  // CurStringPiece - The current string that we are building up as we scan the
  // asm string.
  std::string CurStringPiece;

  bool HasVariants = !C.getTargetInfo().hasNoAsmVariants();

  while (1) {
    // Done with the string?
    if (CurPtr == StrEnd) {
      if (!CurStringPiece.empty())
        Pieces.push_back(AsmStringPiece(CurStringPiece));
      return 0;
    }

    char CurChar = *CurPtr++;
    switch (CurChar) {
    case '$': CurStringPiece += "$$"; continue;
    case '{': CurStringPiece += (HasVariants ? "$(" : "{"); continue;
    case '|': CurStringPiece += (HasVariants ? "$|" : "|"); continue;
    case '}': CurStringPiece += (HasVariants ? "$)" : "}"); continue;
    case '%':
      break;
    default:
      CurStringPiece += CurChar;
      continue;
    }

    // Escaped "%" character in asm string.
    if (CurPtr == StrEnd) {
      // % at end of string is invalid (no escape).
      DiagOffs = CurPtr-StrStart-1;
      return diag::err_asm_invalid_escape;
    }

    char EscapedChar = *CurPtr++;
    if (EscapedChar == '%') {  // %% -> %
      // Escaped percentage sign.
      CurStringPiece += '%';
      continue;
    }

    if (EscapedChar == '=') {  // %= -> Generate an unique ID.
      CurStringPiece += "${:uid}";
      continue;
    }

    // Otherwise, we have an operand.  If we have accumulated a string so far,
    // add it to the Pieces list.
    if (!CurStringPiece.empty()) {
      Pieces.push_back(AsmStringPiece(CurStringPiece));
      CurStringPiece.clear();
    }

    // Handle %x4 and %x[foo] by capturing x as the modifier character.
    char Modifier = '\0';
    if (isalpha(EscapedChar)) {
      if (CurPtr == StrEnd) { // Premature end.
        DiagOffs = CurPtr-StrStart-1;
        return diag::err_asm_invalid_escape;
      }
      Modifier = EscapedChar;
      EscapedChar = *CurPtr++;
    }

    if (isdigit(EscapedChar)) {
      // %n - Assembler operand n
      unsigned N = 0;

      --CurPtr;
      while (CurPtr != StrEnd && isdigit(*CurPtr))
        N = N*10 + ((*CurPtr++)-'0');

      unsigned NumOperands =
        getNumOutputs() + getNumPlusOperands() + getNumInputs();
      if (N >= NumOperands) {
        DiagOffs = CurPtr-StrStart-1;
        return diag::err_asm_invalid_operand_number;
      }

      Pieces.push_back(AsmStringPiece(N, Modifier));
      continue;
    }

    // Handle %[foo], a symbolic operand reference.
    if (EscapedChar == '[') {
      DiagOffs = CurPtr-StrStart-1;

      // Find the ']'.
      const char *NameEnd = (const char*)memchr(CurPtr, ']', StrEnd-CurPtr);
      if (NameEnd == 0)
        return diag::err_asm_unterminated_symbolic_operand_name;
      if (NameEnd == CurPtr)
        return diag::err_asm_empty_symbolic_operand_name;

      StringRef SymbolicName(CurPtr, NameEnd - CurPtr);

      int N = getNamedOperand(SymbolicName);
      if (N == -1) {
        // Verify that an operand with that name exists.
        DiagOffs = CurPtr-StrStart;
        return diag::err_asm_unknown_symbolic_operand_name;
      }
      Pieces.push_back(AsmStringPiece(N, Modifier));

      CurPtr = NameEnd+1;
      continue;
    }

    DiagOffs = CurPtr-StrStart-1;
    return diag::err_asm_invalid_escape;
  }
}

/// Assemble final IR asm string (GCC-style).
std::string GCCAsmStmt::generateAsmString(ASTContext &C) const {
  // Analyze the asm string to decompose it into its pieces.  We know that Sema
  // has already done this, so it is guaranteed to be successful.
  SmallVector<GCCAsmStmt::AsmStringPiece, 4> Pieces;
  unsigned DiagOffs;
  AnalyzeAsmString(Pieces, C, DiagOffs);

  std::string AsmString;
  for (unsigned i = 0, e = Pieces.size(); i != e; ++i) {
    if (Pieces[i].isString())
      AsmString += Pieces[i].getString();
    else if (Pieces[i].getModifier() == '\0')
      AsmString += '$' + llvm::utostr(Pieces[i].getOperandNo());
    else
      AsmString += "${" + llvm::utostr(Pieces[i].getOperandNo()) + ':' +
                   Pieces[i].getModifier() + '}';
  }
  return AsmString;
}

/// Assemble final IR asm string (MS-style).
std::string MSAsmStmt::generateAsmString(ASTContext &C) const {
  // FIXME: This needs to be translated into the IR string representation.
  return AsmStr;
}

Expr *MSAsmStmt::getOutputExpr(unsigned i) {
  return cast<Expr>(Exprs[i]);
}

Expr *MSAsmStmt::getInputExpr(unsigned i) {
  return cast<Expr>(Exprs[i + NumOutputs]);
}
void MSAsmStmt::setInputExpr(unsigned i, Expr *E) {
  Exprs[i + NumOutputs] = E;
}

QualType CXXCatchStmt::getCaughtType() const {
  if (ExceptionDecl)
    return ExceptionDecl->getType();
  return QualType();
}

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

GCCAsmStmt::GCCAsmStmt(ASTContext &C, SourceLocation asmloc, bool issimple,
                       bool isvolatile, unsigned numoutputs, unsigned numinputs,
                       IdentifierInfo **names, StringLiteral **constraints,
                       Expr **exprs, StringLiteral *asmstr,
                       unsigned numclobbers, StringLiteral **clobbers,
                       SourceLocation rparenloc)
  : AsmStmt(GCCAsmStmtClass, asmloc, issimple, isvolatile, numoutputs,
            numinputs, numclobbers), RParenLoc(rparenloc), AsmStr(asmstr) {

  unsigned NumExprs = NumOutputs + NumInputs;

  Names = new (C) IdentifierInfo*[NumExprs];
  std::copy(names, names + NumExprs, Names);

  Exprs = new (C) Stmt*[NumExprs];
  std::copy(exprs, exprs + NumExprs, Exprs);

  Constraints = new (C) StringLiteral*[NumExprs];
  std::copy(constraints, constraints + NumExprs, Constraints);

  Clobbers = new (C) StringLiteral*[NumClobbers];
  std::copy(clobbers, clobbers + NumClobbers, Clobbers);
}

MSAsmStmt::MSAsmStmt(ASTContext &C, SourceLocation asmloc,
                     SourceLocation lbraceloc, bool issimple, bool isvolatile,
                     ArrayRef<Token> asmtoks, unsigned numoutputs,
                     unsigned numinputs, ArrayRef<IdentifierInfo*> names,
                     ArrayRef<StringRef> constraints, ArrayRef<Expr*> exprs,
                     StringRef asmstr, ArrayRef<StringRef> clobbers,
                     SourceLocation endloc)
  : AsmStmt(MSAsmStmtClass, asmloc, issimple, isvolatile, numoutputs,
            numinputs, clobbers.size()), LBraceLoc(lbraceloc),
            EndLoc(endloc), AsmStr(asmstr.str()), NumAsmToks(asmtoks.size()) {

  unsigned NumExprs = NumOutputs + NumInputs;

  Names = new (C) IdentifierInfo*[NumExprs];
  for (unsigned i = 0, e = NumExprs; i != e; ++i)
    Names[i] = names[i];

  Exprs = new (C) Stmt*[NumExprs];
  for (unsigned i = 0, e = NumExprs; i != e; ++i)
    Exprs[i] = exprs[i];

  AsmToks = new (C) Token[NumAsmToks];
  for (unsigned i = 0, e = NumAsmToks; i != e; ++i)
    AsmToks[i] = asmtoks[i];

  Constraints = new (C) StringRef[NumExprs];
  for (unsigned i = 0, e = NumExprs; i != e; ++i) {
    size_t size = constraints[i].size();
    char *dest = new (C) char[size];
    std::strncpy(dest, constraints[i].data(), size); 
    Constraints[i] = StringRef(dest, size);
  }

  Clobbers = new (C) StringRef[NumClobbers];
  for (unsigned i = 0, e = NumClobbers; i != e; ++i) {
    // FIXME: Avoid the allocation/copy if at all possible.
    size_t size = clobbers[i].size();
    char *dest = new (C) char[size];
    std::strncpy(dest, clobbers[i].data(), size); 
    Clobbers[i] = StringRef(dest, size);
  }
}

ObjCForCollectionStmt::ObjCForCollectionStmt(Stmt *Elem, Expr *Collect,
                                             Stmt *Body,  SourceLocation FCL,
                                             SourceLocation RPL)
: Stmt(ObjCForCollectionStmtClass) {
  SubExprs[ELEM] = Elem;
  SubExprs[COLLECTION] = reinterpret_cast<Stmt*>(Collect);
  SubExprs[BODY] = Body;
  ForLoc = FCL;
  RParenLoc = RPL;
}

ObjCAtTryStmt::ObjCAtTryStmt(SourceLocation atTryLoc, Stmt *atTryStmt,
                             Stmt **CatchStmts, unsigned NumCatchStmts,
                             Stmt *atFinallyStmt)
  : Stmt(ObjCAtTryStmtClass), AtTryLoc(atTryLoc),
    NumCatchStmts(NumCatchStmts), HasFinally(atFinallyStmt != 0)
{
  Stmt **Stmts = getStmts();
  Stmts[0] = atTryStmt;
  for (unsigned I = 0; I != NumCatchStmts; ++I)
    Stmts[I + 1] = CatchStmts[I];

  if (HasFinally)
    Stmts[NumCatchStmts + 1] = atFinallyStmt;
}

ObjCAtTryStmt *ObjCAtTryStmt::Create(ASTContext &Context,
                                     SourceLocation atTryLoc,
                                     Stmt *atTryStmt,
                                     Stmt **CatchStmts,
                                     unsigned NumCatchStmts,
                                     Stmt *atFinallyStmt) {
  unsigned Size = sizeof(ObjCAtTryStmt) +
    (1 + NumCatchStmts + (atFinallyStmt != 0)) * sizeof(Stmt *);
  void *Mem = Context.Allocate(Size, llvm::alignOf<ObjCAtTryStmt>());
  return new (Mem) ObjCAtTryStmt(atTryLoc, atTryStmt, CatchStmts, NumCatchStmts,
                                 atFinallyStmt);
}

ObjCAtTryStmt *ObjCAtTryStmt::CreateEmpty(ASTContext &Context,
                                                 unsigned NumCatchStmts,
                                                 bool HasFinally) {
  unsigned Size = sizeof(ObjCAtTryStmt) +
    (1 + NumCatchStmts + HasFinally) * sizeof(Stmt *);
  void *Mem = Context.Allocate(Size, llvm::alignOf<ObjCAtTryStmt>());
  return new (Mem) ObjCAtTryStmt(EmptyShell(), NumCatchStmts, HasFinally);
}

SourceRange ObjCAtTryStmt::getSourceRange() const {
  SourceLocation EndLoc;
  if (HasFinally)
    EndLoc = getFinallyStmt()->getLocEnd();
  else if (NumCatchStmts)
    EndLoc = getCatchStmt(NumCatchStmts - 1)->getLocEnd();
  else
    EndLoc = getTryBody()->getLocEnd();

  return SourceRange(AtTryLoc, EndLoc);
}

CXXTryStmt *CXXTryStmt::Create(ASTContext &C, SourceLocation tryLoc,
                               Stmt *tryBlock, Stmt **handlers,
                               unsigned numHandlers) {
  std::size_t Size = sizeof(CXXTryStmt);
  Size += ((numHandlers + 1) * sizeof(Stmt));

  void *Mem = C.Allocate(Size, llvm::alignOf<CXXTryStmt>());
  return new (Mem) CXXTryStmt(tryLoc, tryBlock, handlers, numHandlers);
}

CXXTryStmt *CXXTryStmt::Create(ASTContext &C, EmptyShell Empty,
                               unsigned numHandlers) {
  std::size_t Size = sizeof(CXXTryStmt);
  Size += ((numHandlers + 1) * sizeof(Stmt));

  void *Mem = C.Allocate(Size, llvm::alignOf<CXXTryStmt>());
  return new (Mem) CXXTryStmt(Empty, numHandlers);
}

CXXTryStmt::CXXTryStmt(SourceLocation tryLoc, Stmt *tryBlock,
                       Stmt **handlers, unsigned numHandlers)
  : Stmt(CXXTryStmtClass), TryLoc(tryLoc), NumHandlers(numHandlers) {
  Stmt **Stmts = reinterpret_cast<Stmt **>(this + 1);
  Stmts[0] = tryBlock;
  std::copy(handlers, handlers + NumHandlers, Stmts + 1);
}

CXXForRangeStmt::CXXForRangeStmt(DeclStmt *Range, DeclStmt *BeginEndStmt,
                                 Expr *Cond, Expr *Inc, DeclStmt *LoopVar,
                                 Stmt *Body, SourceLocation FL,
                                 SourceLocation CL, SourceLocation RPL)
  : Stmt(CXXForRangeStmtClass), ForLoc(FL), ColonLoc(CL), RParenLoc(RPL) {
  SubExprs[RANGE] = Range;
  SubExprs[BEGINEND] = BeginEndStmt;
  SubExprs[COND] = reinterpret_cast<Stmt*>(Cond);
  SubExprs[INC] = reinterpret_cast<Stmt*>(Inc);
  SubExprs[LOOPVAR] = LoopVar;
  SubExprs[BODY] = Body;
}

Expr *CXXForRangeStmt::getRangeInit() {
  DeclStmt *RangeStmt = getRangeStmt();
  VarDecl *RangeDecl = dyn_cast_or_null<VarDecl>(RangeStmt->getSingleDecl());
  assert(RangeDecl &&& "for-range should have a single var decl");
  return RangeDecl->getInit();
}

const Expr *CXXForRangeStmt::getRangeInit() const {
  return const_cast<CXXForRangeStmt*>(this)->getRangeInit();
}

VarDecl *CXXForRangeStmt::getLoopVariable() {
  Decl *LV = cast<DeclStmt>(getLoopVarStmt())->getSingleDecl();
  assert(LV && "No loop variable in CXXForRangeStmt");
  return cast<VarDecl>(LV);
}

const VarDecl *CXXForRangeStmt::getLoopVariable() const {
  return const_cast<CXXForRangeStmt*>(this)->getLoopVariable();
}

IfStmt::IfStmt(ASTContext &C, SourceLocation IL, VarDecl *var, Expr *cond,
               Stmt *then, SourceLocation EL, Stmt *elsev)
  : Stmt(IfStmtClass), IfLoc(IL), ElseLoc(EL)
{
  setConditionVariable(C, var);
  SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
  SubExprs[THEN] = then;
  SubExprs[ELSE] = elsev;
}

VarDecl *IfStmt::getConditionVariable() const {
  if (!SubExprs[VAR])
    return 0;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[VAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void IfStmt::setConditionVariable(ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[VAR] = 0;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[VAR] = new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(),
                                   VarRange.getEnd());
}

ForStmt::ForStmt(ASTContext &C, Stmt *Init, Expr *Cond, VarDecl *condVar,
                 Expr *Inc, Stmt *Body, SourceLocation FL, SourceLocation LP,
                 SourceLocation RP)
  : Stmt(ForStmtClass), ForLoc(FL), LParenLoc(LP), RParenLoc(RP)
{
  SubExprs[INIT] = Init;
  setConditionVariable(C, condVar);
  SubExprs[COND] = reinterpret_cast<Stmt*>(Cond);
  SubExprs[INC] = reinterpret_cast<Stmt*>(Inc);
  SubExprs[BODY] = Body;
}

VarDecl *ForStmt::getConditionVariable() const {
  if (!SubExprs[CONDVAR])
    return 0;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[CONDVAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void ForStmt::setConditionVariable(ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[CONDVAR] = 0;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[CONDVAR] = new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(),
                                       VarRange.getEnd());
}

SwitchStmt::SwitchStmt(ASTContext &C, VarDecl *Var, Expr *cond)
  : Stmt(SwitchStmtClass), FirstCase(0), AllEnumCasesCovered(0)
{
  setConditionVariable(C, Var);
  SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
  SubExprs[BODY] = NULL;
}

VarDecl *SwitchStmt::getConditionVariable() const {
  if (!SubExprs[VAR])
    return 0;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[VAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void SwitchStmt::setConditionVariable(ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[VAR] = 0;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[VAR] = new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(),
                                   VarRange.getEnd());
}

Stmt *SwitchCase::getSubStmt() {
  if (isa<CaseStmt>(this))
    return cast<CaseStmt>(this)->getSubStmt();
  return cast<DefaultStmt>(this)->getSubStmt();
}

WhileStmt::WhileStmt(ASTContext &C, VarDecl *Var, Expr *cond, Stmt *body,
                     SourceLocation WL)
  : Stmt(WhileStmtClass) {
  setConditionVariable(C, Var);
  SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
  SubExprs[BODY] = body;
  WhileLoc = WL;
}

VarDecl *WhileStmt::getConditionVariable() const {
  if (!SubExprs[VAR])
    return 0;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[VAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void WhileStmt::setConditionVariable(ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[VAR] = 0;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[VAR] = new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(),
                                   VarRange.getEnd());
}

// IndirectGotoStmt
LabelDecl *IndirectGotoStmt::getConstantTarget() {
  if (AddrLabelExpr *E =
        dyn_cast<AddrLabelExpr>(getTarget()->IgnoreParenImpCasts()))
    return E->getLabel();
  return 0;
}

// ReturnStmt
const Expr* ReturnStmt::getRetValue() const {
  return cast_or_null<Expr>(RetExpr);
}
Expr* ReturnStmt::getRetValue() {
  return cast_or_null<Expr>(RetExpr);
}

SEHTryStmt::SEHTryStmt(bool IsCXXTry,
                       SourceLocation TryLoc,
                       Stmt *TryBlock,
                       Stmt *Handler)
  : Stmt(SEHTryStmtClass),
    IsCXXTry(IsCXXTry),
    TryLoc(TryLoc)
{
  Children[TRY]     = TryBlock;
  Children[HANDLER] = Handler;
}

SEHTryStmt* SEHTryStmt::Create(ASTContext &C,
                               bool IsCXXTry,
                               SourceLocation TryLoc,
                               Stmt *TryBlock,
                               Stmt *Handler) {
  return new(C) SEHTryStmt(IsCXXTry,TryLoc,TryBlock,Handler);
}

SEHExceptStmt* SEHTryStmt::getExceptHandler() const {
  return dyn_cast<SEHExceptStmt>(getHandler());
}

SEHFinallyStmt* SEHTryStmt::getFinallyHandler() const {
  return dyn_cast<SEHFinallyStmt>(getHandler());
}

SEHExceptStmt::SEHExceptStmt(SourceLocation Loc,
                             Expr *FilterExpr,
                             Stmt *Block)
  : Stmt(SEHExceptStmtClass),
    Loc(Loc)
{
  Children[FILTER_EXPR] = reinterpret_cast<Stmt*>(FilterExpr);
  Children[BLOCK]       = Block;
}

SEHExceptStmt* SEHExceptStmt::Create(ASTContext &C,
                                     SourceLocation Loc,
                                     Expr *FilterExpr,
                                     Stmt *Block) {
  return new(C) SEHExceptStmt(Loc,FilterExpr,Block);
}

SEHFinallyStmt::SEHFinallyStmt(SourceLocation Loc,
                               Stmt *Block)
  : Stmt(SEHFinallyStmtClass),
    Loc(Loc),
    Block(Block)
{}

SEHFinallyStmt* SEHFinallyStmt::Create(ASTContext &C,
                                       SourceLocation Loc,
                                       Stmt *Block) {
  return new(C)SEHFinallyStmt(Loc,Block);
}
