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

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Token.h"
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

void *Stmt::operator new(size_t bytes, const ASTContext& C,
                         unsigned alignment) {
  return ::operator new(bytes, C, alignment);
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
    if (StmtClassInfo[i].Name == nullptr) continue;
    sum += StmtClassInfo[i].Counter;
  }
  llvm::errs() << "  " << sum << " stmts/exprs total.\n";
  sum = 0;
  for (int i = 0; i != Stmt::lastStmtConstant+1; i++) {
    if (StmtClassInfo[i].Name == nullptr) continue;
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

  if (auto *ewc = dyn_cast<ExprWithCleanups>(s))
    s = ewc->getSubExpr();

  if (auto *mte = dyn_cast<MaterializeTemporaryExpr>(s))
    s = mte->GetTemporaryExpr();

  if (auto *bte = dyn_cast<CXXBindTemporaryExpr>(s))
    s = bte->getSubExpr();

  while (auto *ice = dyn_cast<ImplicitCastExpr>(s))
    s = ice->getSubExpr();

  return s;
}

/// \brief Skip no-op (attributed, compound) container stmts and skip captured
/// stmt at the top, if \a IgnoreCaptured is true.
Stmt *Stmt::IgnoreContainers(bool IgnoreCaptured) {
  Stmt *S = this;
  if (IgnoreCaptured)
    if (auto CapS = dyn_cast_or_null<CapturedStmt>(S))
      S = CapS->getCapturedStmt();
  while (true) {
    if (auto AS = dyn_cast_or_null<AttributedStmt>(S))
      S = AS->getSubStmt();
    else if (auto CS = dyn_cast_or_null<CompoundStmt>(S)) {
      if (CS->size() != 1)
        break;
      S = CS->body_back();
    } else
      break;
  }
  return S;
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
  LLVM_ATTRIBUTE_UNUSED
  static inline bad implements_children(children_t Stmt::*) {
    return bad();
  }

  typedef SourceLocation getLocStart_t() const;
  template <class T> good implements_getLocStart(getLocStart_t T::*) {
    return good();
  }
  LLVM_ATTRIBUTE_UNUSED
  static inline bad implements_getLocStart(getLocStart_t Stmt::*) {
    return bad();
  }

  typedef SourceLocation getLocEnd_t() const;
  template <class T> good implements_getLocEnd(getLocEnd_t T::*) {
    return good();
  }
  LLVM_ATTRIBUTE_UNUSED
  static inline bad implements_getLocEnd(getLocEnd_t Stmt::*) {
    return bad();
  }

#define ASSERT_IMPLEMENTS_children(type) \
  (void) is_good(implements_children(&type::children))
#define ASSERT_IMPLEMENTS_getLocStart(type) \
  (void) is_good(implements_getLocStart(&type::getLocStart))
#define ASSERT_IMPLEMENTS_getLocEnd(type) \
  (void) is_good(implements_getLocEnd(&type::getLocEnd))
}

/// Check whether the various Stmt classes implement their member
/// functions.
LLVM_ATTRIBUTE_UNUSED
static inline void check_implementations() {
#define ABSTRACT_STMT(type)
#define STMT(type, base) \
  ASSERT_IMPLEMENTS_children(type); \
  ASSERT_IMPLEMENTS_getLocStart(type); \
  ASSERT_IMPLEMENTS_getLocEnd(type);
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

// Amusing macro metaprogramming hack: check whether a class provides
// a more specific implementation of getSourceRange.
//
// See also Expr.cpp:getExprLoc().
namespace {
  /// This implementation is used when a class provides a custom
  /// implementation of getSourceRange.
  template <class S, class T>
  SourceRange getSourceRangeImpl(const Stmt *stmt,
                                 SourceRange (T::*v)() const) {
    return static_cast<const S*>(stmt)->getSourceRange();
  }

  /// This implementation is used when a class doesn't provide a custom
  /// implementation of getSourceRange.  Overload resolution should pick it over
  /// the implementation above because it's more specialized according to
  /// function template partial ordering.
  template <class S>
  SourceRange getSourceRangeImpl(const Stmt *stmt,
                                 SourceRange (Stmt::*v)() const) {
    return SourceRange(static_cast<const S*>(stmt)->getLocStart(),
                       static_cast<const S*>(stmt)->getLocEnd());
  }
}

SourceRange Stmt::getSourceRange() const {
  switch (getStmtClass()) {
  case Stmt::NoStmtClass: llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base) \
  case Stmt::type##Class: \
    return getSourceRangeImpl<type>(this, &type::getSourceRange);
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind!");
}

SourceLocation Stmt::getLocStart() const {
//  llvm::errs() << "getLocStart() for " << getStmtClassName() << "\n";
  switch (getStmtClass()) {
  case Stmt::NoStmtClass: llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base) \
  case Stmt::type##Class: \
    return static_cast<const type*>(this)->getLocStart();
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
    return static_cast<const type*>(this)->getLocEnd();
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind");
}

CompoundStmt::CompoundStmt(const ASTContext &C, ArrayRef<Stmt*> Stmts,
                           SourceLocation LB, SourceLocation RB)
  : Stmt(CompoundStmtClass), LBraceLoc(LB), RBraceLoc(RB) {
  CompoundStmtBits.NumStmts = Stmts.size();
  assert(CompoundStmtBits.NumStmts == Stmts.size() &&
         "NumStmts doesn't fit in bits of CompoundStmtBits.NumStmts!");

  if (Stmts.size() == 0) {
    Body = nullptr;
    return;
  }

  Body = new (C) Stmt*[Stmts.size()];
  std::copy(Stmts.begin(), Stmts.end(), Body);
}

void CompoundStmt::setStmts(const ASTContext &C, Stmt **Stmts,
                            unsigned NumStmts) {
  if (this->Body)
    C.Deallocate(Body);
  this->CompoundStmtBits.NumStmts = NumStmts;

  Body = new (C) Stmt*[NumStmts];
  memcpy(Body, Stmts, sizeof(Stmt *) * NumStmts);
}

const char *LabelStmt::getName() const {
  return getDecl()->getIdentifier()->getNameStart();
}

AttributedStmt *AttributedStmt::Create(const ASTContext &C, SourceLocation Loc,
                                       ArrayRef<const Attr*> Attrs,
                                       Stmt *SubStmt) {
  assert(!Attrs.empty() && "Attrs should not be empty");
  void *Mem = C.Allocate(sizeof(AttributedStmt) + sizeof(Attr *) * Attrs.size(),
                         llvm::alignOf<AttributedStmt>());
  return new (Mem) AttributedStmt(Loc, Attrs, SubStmt);
}

AttributedStmt *AttributedStmt::CreateEmpty(const ASTContext &C,
                                            unsigned NumAttrs) {
  assert(NumAttrs > 0 && "NumAttrs should be greater than zero");
  void *Mem = C.Allocate(sizeof(AttributedStmt) + sizeof(Attr *) * NumAttrs,
                         llvm::alignOf<AttributedStmt>());
  return new (Mem) AttributedStmt(EmptyShell(), NumAttrs);
}

std::string AsmStmt::generateAsmString(const ASTContext &C) const {
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

char GCCAsmStmt::AsmStringPiece::getModifier() const {
  assert(isOperand() && "Only Operands can have modifiers.");
  return isLetter(Str[0]) ? Str[0] : '\0';
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

void GCCAsmStmt::setOutputsAndInputsAndClobbers(const ASTContext &C,
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
                                const ASTContext &C, unsigned &DiagOffs) const {
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

    // Handle operands that have asmSymbolicName (e.g., %x[foo]) and those that
    // don't (e.g., %x4). 'x' following the '%' is the constraint modifier.

    const char *Begin = CurPtr - 1; // Points to the character following '%'.
    const char *Percent = Begin - 1; // Points to '%'.

    if (isLetter(EscapedChar)) {
      if (CurPtr == StrEnd) { // Premature end.
        DiagOffs = CurPtr-StrStart-1;
        return diag::err_asm_invalid_escape;
      }
      EscapedChar = *CurPtr++;
    }

    const TargetInfo &TI = C.getTargetInfo();
    const SourceManager &SM = C.getSourceManager();
    const LangOptions &LO = C.getLangOpts();

    // Handle operands that don't have asmSymbolicName (e.g., %x4).
    if (isDigit(EscapedChar)) {
      // %n - Assembler operand n
      unsigned N = 0;

      --CurPtr;
      while (CurPtr != StrEnd && isDigit(*CurPtr))
        N = N*10 + ((*CurPtr++)-'0');

      unsigned NumOperands =
        getNumOutputs() + getNumPlusOperands() + getNumInputs();
      if (N >= NumOperands) {
        DiagOffs = CurPtr-StrStart-1;
        return diag::err_asm_invalid_operand_number;
      }

      // Str contains "x4" (Operand without the leading %).
      std::string Str(Begin, CurPtr - Begin);

      // (BeginLoc, EndLoc) represents the range of the operand we are currently
      // processing. Unlike Str, the range includes the leading '%'.
      SourceLocation BeginLoc =
          getAsmString()->getLocationOfByte(Percent - StrStart, SM, LO, TI);
      SourceLocation EndLoc =
          getAsmString()->getLocationOfByte(CurPtr - StrStart, SM, LO, TI);

      Pieces.emplace_back(N, std::move(Str), BeginLoc, EndLoc);
      continue;
    }

    // Handle operands that have asmSymbolicName (e.g., %x[foo]).
    if (EscapedChar == '[') {
      DiagOffs = CurPtr-StrStart-1;

      // Find the ']'.
      const char *NameEnd = (const char*)memchr(CurPtr, ']', StrEnd-CurPtr);
      if (NameEnd == nullptr)
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

      // Str contains "x[foo]" (Operand without the leading %).
      std::string Str(Begin, NameEnd + 1 - Begin);

      // (BeginLoc, EndLoc) represents the range of the operand we are currently
      // processing. Unlike Str, the range includes the leading '%'.
      SourceLocation BeginLoc =
          getAsmString()->getLocationOfByte(Percent - StrStart, SM, LO, TI);
      SourceLocation EndLoc =
          getAsmString()->getLocationOfByte(NameEnd + 1 - StrStart, SM, LO, TI);

      Pieces.emplace_back(N, std::move(Str), BeginLoc, EndLoc);

      CurPtr = NameEnd+1;
      continue;
    }

    DiagOffs = CurPtr-StrStart-1;
    return diag::err_asm_invalid_escape;
  }
}

/// Assemble final IR asm string (GCC-style).
std::string GCCAsmStmt::generateAsmString(const ASTContext &C) const {
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
std::string MSAsmStmt::generateAsmString(const ASTContext &C) const {
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

GCCAsmStmt::GCCAsmStmt(const ASTContext &C, SourceLocation asmloc,
                       bool issimple, bool isvolatile, unsigned numoutputs,
                       unsigned numinputs, IdentifierInfo **names,
                       StringLiteral **constraints, Expr **exprs,
                       StringLiteral *asmstr, unsigned numclobbers,
                       StringLiteral **clobbers, SourceLocation rparenloc)
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

MSAsmStmt::MSAsmStmt(const ASTContext &C, SourceLocation asmloc,
                     SourceLocation lbraceloc, bool issimple, bool isvolatile,
                     ArrayRef<Token> asmtoks, unsigned numoutputs,
                     unsigned numinputs,
                     ArrayRef<StringRef> constraints, ArrayRef<Expr*> exprs,
                     StringRef asmstr, ArrayRef<StringRef> clobbers,
                     SourceLocation endloc)
  : AsmStmt(MSAsmStmtClass, asmloc, issimple, isvolatile, numoutputs,
            numinputs, clobbers.size()), LBraceLoc(lbraceloc),
            EndLoc(endloc), NumAsmToks(asmtoks.size()) {

  initialize(C, asmstr, asmtoks, constraints, exprs, clobbers);
}

static StringRef copyIntoContext(const ASTContext &C, StringRef str) {
  if (str.empty())
    return StringRef();
  size_t size = str.size();
  char *buffer = new (C) char[size];
  memcpy(buffer, str.data(), size);
  return StringRef(buffer, size);
}

void MSAsmStmt::initialize(const ASTContext &C, StringRef asmstr,
                           ArrayRef<Token> asmtoks,
                           ArrayRef<StringRef> constraints,
                           ArrayRef<Expr*> exprs,
                           ArrayRef<StringRef> clobbers) {
  assert(NumAsmToks == asmtoks.size());
  assert(NumClobbers == clobbers.size());

  unsigned NumExprs = exprs.size();
  assert(NumExprs == NumOutputs + NumInputs);
  assert(NumExprs == constraints.size());

  AsmStr = copyIntoContext(C, asmstr);

  Exprs = new (C) Stmt*[NumExprs];
  for (unsigned i = 0, e = NumExprs; i != e; ++i)
    Exprs[i] = exprs[i];

  AsmToks = new (C) Token[NumAsmToks];
  for (unsigned i = 0, e = NumAsmToks; i != e; ++i)
    AsmToks[i] = asmtoks[i];

  Constraints = new (C) StringRef[NumExprs];
  for (unsigned i = 0, e = NumExprs; i != e; ++i) {
    Constraints[i] = copyIntoContext(C, constraints[i]);
  }

  Clobbers = new (C) StringRef[NumClobbers];
  for (unsigned i = 0, e = NumClobbers; i != e; ++i) {
    // FIXME: Avoid the allocation/copy if at all possible.
    Clobbers[i] = copyIntoContext(C, clobbers[i]);
  }
}

ObjCForCollectionStmt::ObjCForCollectionStmt(Stmt *Elem, Expr *Collect,
                                             Stmt *Body,  SourceLocation FCL,
                                             SourceLocation RPL)
: Stmt(ObjCForCollectionStmtClass) {
  SubExprs[ELEM] = Elem;
  SubExprs[COLLECTION] = Collect;
  SubExprs[BODY] = Body;
  ForLoc = FCL;
  RParenLoc = RPL;
}

ObjCAtTryStmt::ObjCAtTryStmt(SourceLocation atTryLoc, Stmt *atTryStmt,
                             Stmt **CatchStmts, unsigned NumCatchStmts,
                             Stmt *atFinallyStmt)
  : Stmt(ObjCAtTryStmtClass), AtTryLoc(atTryLoc),
    NumCatchStmts(NumCatchStmts), HasFinally(atFinallyStmt != nullptr) {
  Stmt **Stmts = getStmts();
  Stmts[0] = atTryStmt;
  for (unsigned I = 0; I != NumCatchStmts; ++I)
    Stmts[I + 1] = CatchStmts[I];

  if (HasFinally)
    Stmts[NumCatchStmts + 1] = atFinallyStmt;
}

ObjCAtTryStmt *ObjCAtTryStmt::Create(const ASTContext &Context,
                                     SourceLocation atTryLoc,
                                     Stmt *atTryStmt,
                                     Stmt **CatchStmts,
                                     unsigned NumCatchStmts,
                                     Stmt *atFinallyStmt) {
  unsigned Size = sizeof(ObjCAtTryStmt) +
    (1 + NumCatchStmts + (atFinallyStmt != nullptr)) * sizeof(Stmt *);
  void *Mem = Context.Allocate(Size, llvm::alignOf<ObjCAtTryStmt>());
  return new (Mem) ObjCAtTryStmt(atTryLoc, atTryStmt, CatchStmts, NumCatchStmts,
                                 atFinallyStmt);
}

ObjCAtTryStmt *ObjCAtTryStmt::CreateEmpty(const ASTContext &Context,
                                          unsigned NumCatchStmts,
                                          bool HasFinally) {
  unsigned Size = sizeof(ObjCAtTryStmt) +
    (1 + NumCatchStmts + HasFinally) * sizeof(Stmt *);
  void *Mem = Context.Allocate(Size, llvm::alignOf<ObjCAtTryStmt>());
  return new (Mem) ObjCAtTryStmt(EmptyShell(), NumCatchStmts, HasFinally);
}

SourceLocation ObjCAtTryStmt::getLocEnd() const {
  if (HasFinally)
    return getFinallyStmt()->getLocEnd();
  if (NumCatchStmts)
    return getCatchStmt(NumCatchStmts - 1)->getLocEnd();
  return getTryBody()->getLocEnd();
}

CXXTryStmt *CXXTryStmt::Create(const ASTContext &C, SourceLocation tryLoc,
                               Stmt *tryBlock, ArrayRef<Stmt*> handlers) {
  std::size_t Size = sizeof(CXXTryStmt);
  Size += ((handlers.size() + 1) * sizeof(Stmt));

  void *Mem = C.Allocate(Size, llvm::alignOf<CXXTryStmt>());
  return new (Mem) CXXTryStmt(tryLoc, tryBlock, handlers);
}

CXXTryStmt *CXXTryStmt::Create(const ASTContext &C, EmptyShell Empty,
                               unsigned numHandlers) {
  std::size_t Size = sizeof(CXXTryStmt);
  Size += ((numHandlers + 1) * sizeof(Stmt));

  void *Mem = C.Allocate(Size, llvm::alignOf<CXXTryStmt>());
  return new (Mem) CXXTryStmt(Empty, numHandlers);
}

CXXTryStmt::CXXTryStmt(SourceLocation tryLoc, Stmt *tryBlock,
                       ArrayRef<Stmt*> handlers)
  : Stmt(CXXTryStmtClass), TryLoc(tryLoc), NumHandlers(handlers.size()) {
  Stmt **Stmts = reinterpret_cast<Stmt **>(this + 1);
  Stmts[0] = tryBlock;
  std::copy(handlers.begin(), handlers.end(), Stmts + 1);
}

CXXForRangeStmt::CXXForRangeStmt(DeclStmt *Range, DeclStmt *BeginEndStmt,
                                 Expr *Cond, Expr *Inc, DeclStmt *LoopVar,
                                 Stmt *Body, SourceLocation FL,
                                 SourceLocation CL, SourceLocation RPL)
  : Stmt(CXXForRangeStmtClass), ForLoc(FL), ColonLoc(CL), RParenLoc(RPL) {
  SubExprs[RANGE] = Range;
  SubExprs[BEGINEND] = BeginEndStmt;
  SubExprs[COND] = Cond;
  SubExprs[INC] = Inc;
  SubExprs[LOOPVAR] = LoopVar;
  SubExprs[BODY] = Body;
}

Expr *CXXForRangeStmt::getRangeInit() {
  DeclStmt *RangeStmt = getRangeStmt();
  VarDecl *RangeDecl = dyn_cast_or_null<VarDecl>(RangeStmt->getSingleDecl());
  assert(RangeDecl && "for-range should have a single var decl");
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

IfStmt::IfStmt(const ASTContext &C, SourceLocation IL, VarDecl *var, Expr *cond,
               Stmt *then, SourceLocation EL, Stmt *elsev)
  : Stmt(IfStmtClass), IfLoc(IL), ElseLoc(EL)
{
  setConditionVariable(C, var);
  SubExprs[COND] = cond;
  SubExprs[THEN] = then;
  SubExprs[ELSE] = elsev;
}

VarDecl *IfStmt::getConditionVariable() const {
  if (!SubExprs[VAR])
    return nullptr;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[VAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void IfStmt::setConditionVariable(const ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[VAR] = nullptr;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[VAR] = new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(),
                                   VarRange.getEnd());
}

ForStmt::ForStmt(const ASTContext &C, Stmt *Init, Expr *Cond, VarDecl *condVar,
                 Expr *Inc, Stmt *Body, SourceLocation FL, SourceLocation LP,
                 SourceLocation RP)
  : Stmt(ForStmtClass), ForLoc(FL), LParenLoc(LP), RParenLoc(RP)
{
  SubExprs[INIT] = Init;
  setConditionVariable(C, condVar);
  SubExprs[COND] = Cond;
  SubExprs[INC] = Inc;
  SubExprs[BODY] = Body;
}

VarDecl *ForStmt::getConditionVariable() const {
  if (!SubExprs[CONDVAR])
    return nullptr;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[CONDVAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void ForStmt::setConditionVariable(const ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[CONDVAR] = nullptr;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[CONDVAR] = new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(),
                                       VarRange.getEnd());
}

SwitchStmt::SwitchStmt(const ASTContext &C, VarDecl *Var, Expr *cond)
    : Stmt(SwitchStmtClass), FirstCase(nullptr, false) {
  setConditionVariable(C, Var);
  SubExprs[COND] = cond;
  SubExprs[BODY] = nullptr;
}

VarDecl *SwitchStmt::getConditionVariable() const {
  if (!SubExprs[VAR])
    return nullptr;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[VAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void SwitchStmt::setConditionVariable(const ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[VAR] = nullptr;
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

WhileStmt::WhileStmt(const ASTContext &C, VarDecl *Var, Expr *cond, Stmt *body,
                     SourceLocation WL)
  : Stmt(WhileStmtClass) {
  setConditionVariable(C, Var);
  SubExprs[COND] = cond;
  SubExprs[BODY] = body;
  WhileLoc = WL;
}

VarDecl *WhileStmt::getConditionVariable() const {
  if (!SubExprs[VAR])
    return nullptr;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[VAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void WhileStmt::setConditionVariable(const ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[VAR] = nullptr;
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
  return nullptr;
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

SEHTryStmt* SEHTryStmt::Create(const ASTContext &C, bool IsCXXTry,
                               SourceLocation TryLoc, Stmt *TryBlock,
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
  Children[FILTER_EXPR] = FilterExpr;
  Children[BLOCK]       = Block;
}

SEHExceptStmt* SEHExceptStmt::Create(const ASTContext &C, SourceLocation Loc,
                                     Expr *FilterExpr, Stmt *Block) {
  return new(C) SEHExceptStmt(Loc,FilterExpr,Block);
}

SEHFinallyStmt::SEHFinallyStmt(SourceLocation Loc,
                               Stmt *Block)
  : Stmt(SEHFinallyStmtClass),
    Loc(Loc),
    Block(Block)
{}

SEHFinallyStmt* SEHFinallyStmt::Create(const ASTContext &C, SourceLocation Loc,
                                       Stmt *Block) {
  return new(C)SEHFinallyStmt(Loc,Block);
}

CapturedStmt::Capture *CapturedStmt::getStoredCaptures() const {
  unsigned Size = sizeof(CapturedStmt) + sizeof(Stmt *) * (NumCaptures + 1);

  // Offset of the first Capture object.
  unsigned FirstCaptureOffset =
    llvm::RoundUpToAlignment(Size, llvm::alignOf<Capture>());

  return reinterpret_cast<Capture *>(
      reinterpret_cast<char *>(const_cast<CapturedStmt *>(this))
      + FirstCaptureOffset);
}

CapturedStmt::CapturedStmt(Stmt *S, CapturedRegionKind Kind,
                           ArrayRef<Capture> Captures,
                           ArrayRef<Expr *> CaptureInits,
                           CapturedDecl *CD,
                           RecordDecl *RD)
  : Stmt(CapturedStmtClass), NumCaptures(Captures.size()),
    CapDeclAndKind(CD, Kind), TheRecordDecl(RD) {
  assert( S && "null captured statement");
  assert(CD && "null captured declaration for captured statement");
  assert(RD && "null record declaration for captured statement");

  // Copy initialization expressions.
  Stmt **Stored = getStoredStmts();
  for (unsigned I = 0, N = NumCaptures; I != N; ++I)
    *Stored++ = CaptureInits[I];

  // Copy the statement being captured.
  *Stored = S;

  // Copy all Capture objects.
  Capture *Buffer = getStoredCaptures();
  std::copy(Captures.begin(), Captures.end(), Buffer);
}

CapturedStmt::CapturedStmt(EmptyShell Empty, unsigned NumCaptures)
  : Stmt(CapturedStmtClass, Empty), NumCaptures(NumCaptures),
    CapDeclAndKind(nullptr, CR_Default), TheRecordDecl(nullptr) {
  getStoredStmts()[NumCaptures] = nullptr;
}

CapturedStmt *CapturedStmt::Create(const ASTContext &Context, Stmt *S,
                                   CapturedRegionKind Kind,
                                   ArrayRef<Capture> Captures,
                                   ArrayRef<Expr *> CaptureInits,
                                   CapturedDecl *CD,
                                   RecordDecl *RD) {
  // The layout is
  //
  // -----------------------------------------------------------
  // | CapturedStmt, Init, ..., Init, S, Capture, ..., Capture |
  // ----------------^-------------------^----------------------
  //                 getStoredStmts()    getStoredCaptures()
  //
  // where S is the statement being captured.
  //
  assert(CaptureInits.size() == Captures.size() && "wrong number of arguments");

  unsigned Size = sizeof(CapturedStmt) + sizeof(Stmt *) * (Captures.size() + 1);
  if (!Captures.empty()) {
    // Realign for the following Capture array.
    Size = llvm::RoundUpToAlignment(Size, llvm::alignOf<Capture>());
    Size += sizeof(Capture) * Captures.size();
  }

  void *Mem = Context.Allocate(Size);
  return new (Mem) CapturedStmt(S, Kind, Captures, CaptureInits, CD, RD);
}

CapturedStmt *CapturedStmt::CreateDeserialized(const ASTContext &Context,
                                               unsigned NumCaptures) {
  unsigned Size = sizeof(CapturedStmt) + sizeof(Stmt *) * (NumCaptures + 1);
  if (NumCaptures > 0) {
    // Realign for the following Capture array.
    Size = llvm::RoundUpToAlignment(Size, llvm::alignOf<Capture>());
    Size += sizeof(Capture) * NumCaptures;
  }

  void *Mem = Context.Allocate(Size);
  return new (Mem) CapturedStmt(EmptyShell(), NumCaptures);
}

Stmt::child_range CapturedStmt::children() {
  // Children are captured field initilizers.
  return child_range(getStoredStmts(), getStoredStmts() + NumCaptures);
}

bool CapturedStmt::capturesVariable(const VarDecl *Var) const {
  for (const auto &I : captures()) {
    if (!I.capturesVariable())
      continue;

    // This does not handle variable redeclarations. This should be
    // extended to capture variables with redeclarations, for example
    // a thread-private variable in OpenMP.
    if (I.getCapturedVar() == Var)
      return true;
  }

  return false;
}

StmtRange OMPClause::children() {
  switch(getClauseKind()) {
  default : break;
#define OPENMP_CLAUSE(Name, Class)                                       \
  case OMPC_ ## Name : return static_cast<Class *>(this)->children();
#include "clang/Basic/OpenMPKinds.def"
  }
  llvm_unreachable("unknown OMPClause");
}

void OMPPrivateClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

OMPPrivateClause *
OMPPrivateClause::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation LParenLoc, SourceLocation EndLoc,
                         ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL) {
  // Allocate space for private variables and initializer expressions.
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         2 * sizeof(Expr *) * VL.size());
  OMPPrivateClause *Clause =
      new (Mem) OMPPrivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  return Clause;
}

OMPPrivateClause *OMPPrivateClause::CreateEmpty(const ASTContext &C,
                                                unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         2 * sizeof(Expr *) * N);
  return new (Mem) OMPPrivateClause(N);
}

void OMPFirstprivateClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

void OMPFirstprivateClause::setInits(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), getPrivateCopies().end());
}

OMPFirstprivateClause *
OMPFirstprivateClause::Create(const ASTContext &C, SourceLocation StartLoc,
                              SourceLocation LParenLoc, SourceLocation EndLoc,
                              ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL,
                              ArrayRef<Expr *> InitVL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFirstprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         3 * sizeof(Expr *) * VL.size());
  OMPFirstprivateClause *Clause =
      new (Mem) OMPFirstprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  Clause->setInits(InitVL);
  return Clause;
}

OMPFirstprivateClause *OMPFirstprivateClause::CreateEmpty(const ASTContext &C,
                                                          unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFirstprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         3 * sizeof(Expr *) * N);
  return new (Mem) OMPFirstprivateClause(N);
}

void OMPLastprivateClause::setPrivateCopies(ArrayRef<Expr *> PrivateCopies) {
  assert(PrivateCopies.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(PrivateCopies.begin(), PrivateCopies.end(), varlist_end());
}

void OMPLastprivateClause::setSourceExprs(ArrayRef<Expr *> SrcExprs) {
  assert(SrcExprs.size() == varlist_size() && "Number of source expressions is "
                                              "not the same as the "
                                              "preallocated buffer");
  std::copy(SrcExprs.begin(), SrcExprs.end(), getPrivateCopies().end());
}

void OMPLastprivateClause::setDestinationExprs(ArrayRef<Expr *> DstExprs) {
  assert(DstExprs.size() == varlist_size() && "Number of destination "
                                              "expressions is not the same as "
                                              "the preallocated buffer");
  std::copy(DstExprs.begin(), DstExprs.end(), getSourceExprs().end());
}

void OMPLastprivateClause::setAssignmentOps(ArrayRef<Expr *> AssignmentOps) {
  assert(AssignmentOps.size() == varlist_size() &&
         "Number of assignment expressions is not the same as the preallocated "
         "buffer");
  std::copy(AssignmentOps.begin(), AssignmentOps.end(),
            getDestinationExprs().end());
}

OMPLastprivateClause *OMPLastprivateClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLastprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         5 * sizeof(Expr *) * VL.size());
  OMPLastprivateClause *Clause =
      new (Mem) OMPLastprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  return Clause;
}

OMPLastprivateClause *OMPLastprivateClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLastprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         5 * sizeof(Expr *) * N);
  return new (Mem) OMPLastprivateClause(N);
}

OMPSharedClause *OMPSharedClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSharedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size());
  OMPSharedClause *Clause = new (Mem) OMPSharedClause(StartLoc, LParenLoc,
                                                      EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OMPSharedClause *OMPSharedClause::CreateEmpty(const ASTContext &C,
                                              unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSharedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N);
  return new (Mem) OMPSharedClause(N);
}

void OMPLinearClause::setInits(ArrayRef<Expr *> IL) {
  assert(IL.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(IL.begin(), IL.end(), varlist_end());
}

void OMPLinearClause::setUpdates(ArrayRef<Expr *> UL) {
  assert(UL.size() == varlist_size() &&
         "Number of updates is not the same as the preallocated buffer");
  std::copy(UL.begin(), UL.end(), getInits().end());
}

void OMPLinearClause::setFinals(ArrayRef<Expr *> FL) {
  assert(FL.size() == varlist_size() &&
         "Number of final updates is not the same as the preallocated buffer");
  std::copy(FL.begin(), FL.end(), getUpdates().end());
}

OMPLinearClause *
OMPLinearClause::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation LParenLoc, SourceLocation ColonLoc,
                        SourceLocation EndLoc, ArrayRef<Expr *> VL,
                        ArrayRef<Expr *> IL, Expr *Step, Expr *CalcStep) {
  // Allocate space for 4 lists (Vars, Inits, Updates, Finals) and 2 expressions
  // (Step and CalcStep).
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLinearClause),
                                                  llvm::alignOf<Expr *>()) +
                         (4 * VL.size() + 2) * sizeof(Expr *));
  OMPLinearClause *Clause = new (Mem)
      OMPLinearClause(StartLoc, LParenLoc, ColonLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setInits(IL);
  // Fill update and final expressions with zeroes, they are provided later,
  // after the directive construction.
  std::fill(Clause->getInits().end(), Clause->getInits().end() + VL.size(),
            nullptr);
  std::fill(Clause->getUpdates().end(), Clause->getUpdates().end() + VL.size(),
            nullptr);
  Clause->setStep(Step);
  Clause->setCalcStep(CalcStep);
  return Clause;
}

OMPLinearClause *OMPLinearClause::CreateEmpty(const ASTContext &C,
                                              unsigned NumVars) {
  // Allocate space for 4 lists (Vars, Inits, Updates, Finals) and 2 expressions
  // (Step and CalcStep).
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLinearClause),
                                                  llvm::alignOf<Expr *>()) +
                         (4 * NumVars + 2) * sizeof(Expr *));
  return new (Mem) OMPLinearClause(NumVars);
}

OMPAlignedClause *
OMPAlignedClause::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation LParenLoc, SourceLocation ColonLoc,
                         SourceLocation EndLoc, ArrayRef<Expr *> VL, Expr *A) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPAlignedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * (VL.size() + 1));
  OMPAlignedClause *Clause = new (Mem)
      OMPAlignedClause(StartLoc, LParenLoc, ColonLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setAlignment(A);
  return Clause;
}

OMPAlignedClause *OMPAlignedClause::CreateEmpty(const ASTContext &C,
                                                unsigned NumVars) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPAlignedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * (NumVars + 1));
  return new (Mem) OMPAlignedClause(NumVars);
}

void OMPCopyinClause::setSourceExprs(ArrayRef<Expr *> SrcExprs) {
  assert(SrcExprs.size() == varlist_size() && "Number of source expressions is "
                                              "not the same as the "
                                              "preallocated buffer");
  std::copy(SrcExprs.begin(), SrcExprs.end(), varlist_end());
}

void OMPCopyinClause::setDestinationExprs(ArrayRef<Expr *> DstExprs) {
  assert(DstExprs.size() == varlist_size() && "Number of destination "
                                              "expressions is not the same as "
                                              "the preallocated buffer");
  std::copy(DstExprs.begin(), DstExprs.end(), getSourceExprs().end());
}

void OMPCopyinClause::setAssignmentOps(ArrayRef<Expr *> AssignmentOps) {
  assert(AssignmentOps.size() == varlist_size() &&
         "Number of assignment expressions is not the same as the preallocated "
         "buffer");
  std::copy(AssignmentOps.begin(), AssignmentOps.end(),
            getDestinationExprs().end());
}

OMPCopyinClause *OMPCopyinClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyinClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * VL.size());
  OMPCopyinClause *Clause = new (Mem) OMPCopyinClause(StartLoc, LParenLoc,
                                                      EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  return Clause;
}

OMPCopyinClause *OMPCopyinClause::CreateEmpty(const ASTContext &C,
                                              unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyinClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * N);
  return new (Mem) OMPCopyinClause(N);
}

void OMPCopyprivateClause::setSourceExprs(ArrayRef<Expr *> SrcExprs) {
  assert(SrcExprs.size() == varlist_size() && "Number of source expressions is "
                                              "not the same as the "
                                              "preallocated buffer");
  std::copy(SrcExprs.begin(), SrcExprs.end(), varlist_end());
}

void OMPCopyprivateClause::setDestinationExprs(ArrayRef<Expr *> DstExprs) {
  assert(DstExprs.size() == varlist_size() && "Number of destination "
                                              "expressions is not the same as "
                                              "the preallocated buffer");
  std::copy(DstExprs.begin(), DstExprs.end(), getSourceExprs().end());
}

void OMPCopyprivateClause::setAssignmentOps(ArrayRef<Expr *> AssignmentOps) {
  assert(AssignmentOps.size() == varlist_size() &&
         "Number of assignment expressions is not the same as the preallocated "
         "buffer");
  std::copy(AssignmentOps.begin(), AssignmentOps.end(),
            getDestinationExprs().end());
}

OMPCopyprivateClause *OMPCopyprivateClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * VL.size());
  OMPCopyprivateClause *Clause =
      new (Mem) OMPCopyprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  return Clause;
}

OMPCopyprivateClause *OMPCopyprivateClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyprivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * N);
  return new (Mem) OMPCopyprivateClause(N);
}

void OMPExecutableDirective::setClauses(ArrayRef<OMPClause *> Clauses) {
  assert(Clauses.size() == getNumClauses() &&
         "Number of clauses is not the same as the preallocated buffer");
  std::copy(Clauses.begin(), Clauses.end(), getClauses().begin());
}

void OMPLoopDirective::setCounters(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of loop counters is not the same as the collapsed number");
  std::copy(A.begin(), A.end(), getCounters().begin());
}

void OMPLoopDirective::setInits(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of counter inits is not the same as the collapsed number");
  std::copy(A.begin(), A.end(), getInits().begin());
}

void OMPLoopDirective::setUpdates(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of counter updates is not the same as the collapsed number");
  std::copy(A.begin(), A.end(), getUpdates().begin());
}

void OMPLoopDirective::setFinals(ArrayRef<Expr *> A) {
  assert(A.size() == getCollapsedNumber() &&
         "Number of counter finals is not the same as the collapsed number");
  std::copy(A.begin(), A.end(), getFinals().begin());
}

void OMPReductionClause::setLHSExprs(ArrayRef<Expr *> LHSExprs) {
  assert(
      LHSExprs.size() == varlist_size() &&
      "Number of LHS expressions is not the same as the preallocated buffer");
  std::copy(LHSExprs.begin(), LHSExprs.end(), varlist_end());
}

void OMPReductionClause::setRHSExprs(ArrayRef<Expr *> RHSExprs) {
  assert(
      RHSExprs.size() == varlist_size() &&
      "Number of RHS expressions is not the same as the preallocated buffer");
  std::copy(RHSExprs.begin(), RHSExprs.end(), getLHSExprs().end());
}

void OMPReductionClause::setReductionOps(ArrayRef<Expr *> ReductionOps) {
  assert(ReductionOps.size() == varlist_size() && "Number of reduction "
                                                  "expressions is not the same "
                                                  "as the preallocated buffer");
  std::copy(ReductionOps.begin(), ReductionOps.end(), getRHSExprs().end());
}

OMPReductionClause *OMPReductionClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VL,
    NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
    ArrayRef<Expr *> LHSExprs, ArrayRef<Expr *> RHSExprs,
    ArrayRef<Expr *> ReductionOps) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPReductionClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * VL.size());
  OMPReductionClause *Clause = new (Mem) OMPReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo);
  Clause->setVarRefs(VL);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  return Clause;
}

OMPReductionClause *OMPReductionClause::CreateEmpty(const ASTContext &C,
                                                    unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPReductionClause),
                                                  llvm::alignOf<Expr *>()) +
                         4 * sizeof(Expr *) * N);
  return new (Mem) OMPReductionClause(N);
}

OMPFlushClause *OMPFlushClause::Create(const ASTContext &C,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc,
                                       ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFlushClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size());
  OMPFlushClause *Clause =
      new (Mem) OMPFlushClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OMPFlushClause *OMPFlushClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFlushClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N);
  return new (Mem) OMPFlushClause(N);
}

OMPDependClause *
OMPDependClause::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation LParenLoc, SourceLocation EndLoc,
                        OpenMPDependClauseKind DepKind, SourceLocation DepLoc,
                        SourceLocation ColonLoc, ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDependClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size());
  OMPDependClause *Clause =
      new (Mem) OMPDependClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setDependencyKind(DepKind);
  Clause->setDependencyLoc(DepLoc);
  Clause->setColonLoc(ColonLoc);
  return Clause;
}

OMPDependClause *OMPDependClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDependClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N);
  return new (Mem) OMPDependClause(N);
}

const OMPClause *
OMPExecutableDirective::getSingleClause(OpenMPClauseKind K) const {
  auto &&I = getClausesOfKind(K);

  if (I) {
    auto *Clause = *I;
    assert(!++I && "There are at least 2 clauses of the specified kind");
    return Clause;
  }
  return nullptr;
}

OMPParallelDirective *OMPParallelDirective::Create(
                                              const ASTContext &C,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc,
                                              ArrayRef<OMPClause *> Clauses,
                                              Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                         sizeof(Stmt *));
  OMPParallelDirective *Dir = new (Mem) OMPParallelDirective(StartLoc, EndLoc,
                                                             Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPParallelDirective *OMPParallelDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                         sizeof(Stmt *));
  return new (Mem) OMPParallelDirective(NumClauses);
}

OMPSimdDirective *
OMPSimdDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, unsigned CollapsedNum,
                         ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
                         const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_simd));
  OMPSimdDirective *Dir = new (Mem)
      OMPSimdDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setCounters(Exprs.Counters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPSimdDirective *OMPSimdDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                unsigned CollapsedNum,
                                                EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_simd));
  return new (Mem) OMPSimdDirective(CollapsedNum, NumClauses);
}

OMPForDirective *
OMPForDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation EndLoc, unsigned CollapsedNum,
                        ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
                        const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPForDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_for));
  OMPForDirective *Dir =
      new (Mem) OMPForDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPForDirective *OMPForDirective::CreateEmpty(const ASTContext &C,
                                              unsigned NumClauses,
                                              unsigned CollapsedNum,
                                              EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPForDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_for));
  return new (Mem) OMPForDirective(CollapsedNum, NumClauses);
}

OMPForSimdDirective *
OMPForSimdDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                            SourceLocation EndLoc, unsigned CollapsedNum,
                            ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
                            const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPForSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_for_simd));
  OMPForSimdDirective *Dir = new (Mem)
      OMPForSimdDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPForSimdDirective *OMPForSimdDirective::CreateEmpty(const ASTContext &C,
                                                      unsigned NumClauses,
                                                      unsigned CollapsedNum,
                                                      EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPForSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                 sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_for_simd));
  return new (Mem) OMPForSimdDirective(CollapsedNum, NumClauses);
}

OMPSectionsDirective *OMPSectionsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSectionsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPSectionsDirective *Dir =
      new (Mem) OMPSectionsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPSectionsDirective *OMPSectionsDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSectionsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPSectionsDirective(NumClauses);
}

OMPSectionDirective *OMPSectionDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc,
                                                 Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSectionDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPSectionDirective *Dir = new (Mem) OMPSectionDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPSectionDirective *OMPSectionDirective::CreateEmpty(const ASTContext &C,
                                                      EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSectionDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPSectionDirective();
}

OMPSingleDirective *OMPSingleDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               ArrayRef<OMPClause *> Clauses,
                                               Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSingleDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPSingleDirective *Dir =
      new (Mem) OMPSingleDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPSingleDirective *OMPSingleDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPSingleDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPSingleDirective(NumClauses);
}

OMPMasterDirective *OMPMasterDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPMasterDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPMasterDirective *Dir = new (Mem) OMPMasterDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPMasterDirective *OMPMasterDirective::CreateEmpty(const ASTContext &C,
                                                    EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPMasterDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPMasterDirective();
}

OMPCriticalDirective *OMPCriticalDirective::Create(
    const ASTContext &C, const DeclarationNameInfo &Name,
    SourceLocation StartLoc, SourceLocation EndLoc, Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPCriticalDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPCriticalDirective *Dir =
      new (Mem) OMPCriticalDirective(Name, StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPCriticalDirective *OMPCriticalDirective::CreateEmpty(const ASTContext &C,
                                                        EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPCriticalDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPCriticalDirective();
}

OMPParallelForDirective *OMPParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelForDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_parallel_for));
  OMPParallelForDirective *Dir = new (Mem)
      OMPParallelForDirective(StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPParallelForDirective *
OMPParallelForDirective::CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                     unsigned CollapsedNum, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelForDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses +
                         sizeof(Stmt *) *
                             numLoopChildren(CollapsedNum, OMPD_parallel_for));
  return new (Mem) OMPParallelForDirective(CollapsedNum, NumClauses);
}

OMPParallelForSimdDirective *OMPParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
    const HelperExprs &Exprs) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelForSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * Clauses.size() +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_parallel_for_simd));
  OMPParallelForSimdDirective *Dir = new (Mem) OMPParallelForSimdDirective(
      StartLoc, EndLoc, CollapsedNum, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setIterationVariable(Exprs.IterationVarRef);
  Dir->setLastIteration(Exprs.LastIteration);
  Dir->setCalcLastIteration(Exprs.CalcLastIteration);
  Dir->setPreCond(Exprs.PreCond);
  Dir->setCond(Exprs.Cond);
  Dir->setInit(Exprs.Init);
  Dir->setInc(Exprs.Inc);
  Dir->setIsLastIterVariable(Exprs.IL);
  Dir->setLowerBoundVariable(Exprs.LB);
  Dir->setUpperBoundVariable(Exprs.UB);
  Dir->setStrideVariable(Exprs.ST);
  Dir->setEnsureUpperBound(Exprs.EUB);
  Dir->setNextLowerBound(Exprs.NLB);
  Dir->setNextUpperBound(Exprs.NUB);
  Dir->setCounters(Exprs.Counters);
  Dir->setInits(Exprs.Inits);
  Dir->setUpdates(Exprs.Updates);
  Dir->setFinals(Exprs.Finals);
  return Dir;
}

OMPParallelForSimdDirective *
OMPParallelForSimdDirective::CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses,
                                         unsigned CollapsedNum, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelForSimdDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(
      Size + sizeof(OMPClause *) * NumClauses +
      sizeof(Stmt *) * numLoopChildren(CollapsedNum, OMPD_parallel_for_simd));
  return new (Mem) OMPParallelForSimdDirective(CollapsedNum, NumClauses);
}

OMPParallelSectionsDirective *OMPParallelSectionsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelSectionsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPParallelSectionsDirective *Dir =
      new (Mem) OMPParallelSectionsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPParallelSectionsDirective *
OMPParallelSectionsDirective::CreateEmpty(const ASTContext &C,
                                          unsigned NumClauses, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPParallelSectionsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPParallelSectionsDirective(NumClauses);
}

OMPTaskDirective *OMPTaskDirective::Create(const ASTContext &C,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc,
                                           ArrayRef<OMPClause *> Clauses,
                                           Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTaskDirective *Dir =
      new (Mem) OMPTaskDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTaskDirective *OMPTaskDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTaskDirective(NumClauses);
}

OMPTaskyieldDirective *OMPTaskyieldDirective::Create(const ASTContext &C,
                                                     SourceLocation StartLoc,
                                                     SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OMPTaskyieldDirective));
  OMPTaskyieldDirective *Dir =
      new (Mem) OMPTaskyieldDirective(StartLoc, EndLoc);
  return Dir;
}

OMPTaskyieldDirective *OMPTaskyieldDirective::CreateEmpty(const ASTContext &C,
                                                          EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPTaskyieldDirective));
  return new (Mem) OMPTaskyieldDirective();
}

OMPBarrierDirective *OMPBarrierDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OMPBarrierDirective));
  OMPBarrierDirective *Dir = new (Mem) OMPBarrierDirective(StartLoc, EndLoc);
  return Dir;
}

OMPBarrierDirective *OMPBarrierDirective::CreateEmpty(const ASTContext &C,
                                                      EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPBarrierDirective));
  return new (Mem) OMPBarrierDirective();
}

OMPTaskwaitDirective *OMPTaskwaitDirective::Create(const ASTContext &C,
                                                   SourceLocation StartLoc,
                                                   SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OMPTaskwaitDirective));
  OMPTaskwaitDirective *Dir = new (Mem) OMPTaskwaitDirective(StartLoc, EndLoc);
  return Dir;
}

OMPTaskwaitDirective *OMPTaskwaitDirective::CreateEmpty(const ASTContext &C,
                                                        EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPTaskwaitDirective));
  return new (Mem) OMPTaskwaitDirective();
}

OMPTaskgroupDirective *OMPTaskgroupDirective::Create(const ASTContext &C,
                                                     SourceLocation StartLoc,
                                                     SourceLocation EndLoc,
                                                     Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskgroupDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPTaskgroupDirective *Dir =
      new (Mem) OMPTaskgroupDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTaskgroupDirective *OMPTaskgroupDirective::CreateEmpty(const ASTContext &C,
                                                          EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTaskgroupDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPTaskgroupDirective();
}

OMPCancellationPointDirective *OMPCancellationPointDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    OpenMPDirectiveKind CancelRegion) {
  unsigned Size = llvm::RoundUpToAlignment(
      sizeof(OMPCancellationPointDirective), llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size);
  OMPCancellationPointDirective *Dir =
      new (Mem) OMPCancellationPointDirective(StartLoc, EndLoc);
  Dir->setCancelRegion(CancelRegion);
  return Dir;
}

OMPCancellationPointDirective *
OMPCancellationPointDirective::CreateEmpty(const ASTContext &C, EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(
      sizeof(OMPCancellationPointDirective), llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size);
  return new (Mem) OMPCancellationPointDirective();
}

OMPCancelDirective *
OMPCancelDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                           SourceLocation EndLoc,
                           OpenMPDirectiveKind CancelRegion) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPCancelDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size);
  OMPCancelDirective *Dir = new (Mem) OMPCancelDirective(StartLoc, EndLoc);
  Dir->setCancelRegion(CancelRegion);
  return Dir;
}

OMPCancelDirective *OMPCancelDirective::CreateEmpty(const ASTContext &C,
                                                    EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPCancelDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size);
  return new (Mem) OMPCancelDirective();
}

OMPFlushDirective *OMPFlushDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<OMPClause *> Clauses) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPFlushDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size());
  OMPFlushDirective *Dir =
      new (Mem) OMPFlushDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  return Dir;
}

OMPFlushDirective *OMPFlushDirective::CreateEmpty(const ASTContext &C,
                                                  unsigned NumClauses,
                                                  EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPFlushDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * NumClauses);
  return new (Mem) OMPFlushDirective(NumClauses);
}

OMPOrderedDirective *OMPOrderedDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc,
                                                 Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPOrderedDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  OMPOrderedDirective *Dir = new (Mem) OMPOrderedDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPOrderedDirective *OMPOrderedDirective::CreateEmpty(const ASTContext &C,
                                                      EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPOrderedDirective),
                                           llvm::alignOf<Stmt *>());
  void *Mem = C.Allocate(Size + sizeof(Stmt *));
  return new (Mem) OMPOrderedDirective();
}

OMPAtomicDirective *OMPAtomicDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *X, Expr *V,
    Expr *E, Expr *UE, bool IsXLHSInRHSPart, bool IsPostfixUpdate) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPAtomicDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem = C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() +
                         5 * sizeof(Stmt *));
  OMPAtomicDirective *Dir =
      new (Mem) OMPAtomicDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setX(X);
  Dir->setV(V);
  Dir->setExpr(E);
  Dir->setUpdateExpr(UE);
  Dir->IsXLHSInRHSPart = IsXLHSInRHSPart;
  Dir->IsPostfixUpdate = IsPostfixUpdate;
  return Dir;
}

OMPAtomicDirective *OMPAtomicDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPAtomicDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + 5 * sizeof(Stmt *));
  return new (Mem) OMPAtomicDirective(NumClauses);
}

OMPTargetDirective *OMPTargetDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               ArrayRef<OMPClause *> Clauses,
                                               Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTargetDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetDirective *Dir =
      new (Mem) OMPTargetDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetDirective *OMPTargetDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned NumClauses,
                                                    EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTargetDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTargetDirective(NumClauses);
}

OMPTeamsDirective *OMPTeamsDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<OMPClause *> Clauses,
                                             Stmt *AssociatedStmt) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTeamsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTeamsDirective *Dir =
      new (Mem) OMPTeamsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTeamsDirective *OMPTeamsDirective::CreateEmpty(const ASTContext &C,
                                                  unsigned NumClauses,
                                                  EmptyShell) {
  unsigned Size = llvm::RoundUpToAlignment(sizeof(OMPTeamsDirective),
                                           llvm::alignOf<OMPClause *>());
  void *Mem =
      C.Allocate(Size + sizeof(OMPClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OMPTeamsDirective(NumClauses);
}

