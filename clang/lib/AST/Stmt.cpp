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
#include "clang/AST/ExprOpenMP.h"
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

void CompoundStmt::setStmts(const ASTContext &C, ArrayRef<Stmt *> Stmts) {
  if (Body)
    C.Deallocate(Body);
  CompoundStmtBits.NumStmts = Stmts.size();
  assert(CompoundStmtBits.NumStmts == Stmts.size() &&
         "NumStmts doesn't fit in bits of CompoundStmtBits.NumStmts!");

  Body = new (C) Stmt*[Stmts.size()];
  std::copy(Stmts.begin(), Stmts.end(), Body);
}

const char *LabelStmt::getName() const {
  return getDecl()->getIdentifier()->getNameStart();
}

AttributedStmt *AttributedStmt::Create(const ASTContext &C, SourceLocation Loc,
                                       ArrayRef<const Attr*> Attrs,
                                       Stmt *SubStmt) {
  assert(!Attrs.empty() && "Attrs should not be empty");
  void *Mem = C.Allocate(sizeof(AttributedStmt) + sizeof(Attr *) * Attrs.size(),
                         alignof(AttributedStmt));
  return new (Mem) AttributedStmt(Loc, Attrs, SubStmt);
}

AttributedStmt *AttributedStmt::CreateEmpty(const ASTContext &C,
                                            unsigned NumAttrs) {
  assert(NumAttrs > 0 && "NumAttrs should be greater than zero");
  void *Mem = C.Allocate(sizeof(AttributedStmt) + sizeof(Attr *) * NumAttrs,
                         alignof(AttributedStmt));
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

  unsigned LastAsmStringToken = 0;
  unsigned LastAsmStringOffset = 0;

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
    // Handle escaped char and continue looping over the asm string.
    char EscapedChar = *CurPtr++;
    switch (EscapedChar) {
    default:
      break;
    case '%': // %% -> %
    case '{': // %{ -> {
    case '}': // %} -> }
      CurStringPiece += EscapedChar;
      continue;
    case '=': // %= -> Generate a unique ID.
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
      SourceLocation BeginLoc = getAsmString()->getLocationOfByte(
          Percent - StrStart, SM, LO, TI, &LastAsmStringToken,
          &LastAsmStringOffset);
      SourceLocation EndLoc = getAsmString()->getLocationOfByte(
          CurPtr - StrStart, SM, LO, TI, &LastAsmStringToken,
          &LastAsmStringOffset);

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
      SourceLocation BeginLoc = getAsmString()->getLocationOfByte(
          Percent - StrStart, SM, LO, TI, &LastAsmStringToken,
          &LastAsmStringOffset);
      SourceLocation EndLoc = getAsmString()->getLocationOfByte(
          NameEnd + 1 - StrStart, SM, LO, TI, &LastAsmStringToken,
          &LastAsmStringOffset);

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
  return str.copy(C);
}

void MSAsmStmt::initialize(const ASTContext &C, StringRef asmstr,
                           ArrayRef<Token> asmtoks,
                           ArrayRef<StringRef> constraints,
                           ArrayRef<Expr*> exprs,
                           ArrayRef<StringRef> clobbers) {
  assert(NumAsmToks == asmtoks.size());
  assert(NumClobbers == clobbers.size());

  assert(exprs.size() == NumOutputs + NumInputs);
  assert(exprs.size() == constraints.size());

  AsmStr = copyIntoContext(C, asmstr);

  Exprs = new (C) Stmt*[exprs.size()];
  std::copy(exprs.begin(), exprs.end(), Exprs);

  AsmToks = new (C) Token[asmtoks.size()];
  std::copy(asmtoks.begin(), asmtoks.end(), AsmToks);

  Constraints = new (C) StringRef[exprs.size()];
  std::transform(constraints.begin(), constraints.end(), Constraints,
                 [&](StringRef Constraint) {
                   return copyIntoContext(C, Constraint);
                 });

  Clobbers = new (C) StringRef[NumClobbers];
  // FIXME: Avoid the allocation/copy if at all possible.
  std::transform(clobbers.begin(), clobbers.end(), Clobbers,
                 [&](StringRef Clobber) {
                   return copyIntoContext(C, Clobber);
                 });
}

IfStmt::IfStmt(const ASTContext &C, SourceLocation IL, bool IsConstexpr,
               Stmt *init, VarDecl *var, Expr *cond, Stmt *then,
               SourceLocation EL, Stmt *elsev)
    : Stmt(IfStmtClass), IfLoc(IL), ElseLoc(EL) {
  setConstexpr(IsConstexpr);
  setConditionVariable(C, var);
  SubExprs[INIT] = init;
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

bool IfStmt::isObjCAvailabilityCheck() const {
  return isa<ObjCAvailabilityCheckExpr>(SubExprs[COND]);
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

SwitchStmt::SwitchStmt(const ASTContext &C, Stmt *init, VarDecl *Var,
                       Expr *cond)
    : Stmt(SwitchStmtClass), FirstCase(nullptr, false) {
  setConditionVariable(C, Var);
  SubExprs[INIT] = init;
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

CapturedStmt::Capture::Capture(SourceLocation Loc, VariableCaptureKind Kind,
                               VarDecl *Var)
    : VarAndKind(Var, Kind), Loc(Loc) {
  switch (Kind) {
  case VCK_This:
    assert(!Var && "'this' capture cannot have a variable!");
    break;
  case VCK_ByRef:
    assert(Var && "capturing by reference must have a variable!");
    break;
  case VCK_ByCopy:
    assert(Var && "capturing by copy must have a variable!");
    assert(
        (Var->getType()->isScalarType() || (Var->getType()->isReferenceType() &&
                                            Var->getType()
                                                ->castAs<ReferenceType>()
                                                ->getPointeeType()
                                                ->isScalarType())) &&
        "captures by copy are expected to have a scalar type!");
    break;
  case VCK_VLAType:
    assert(!Var &&
           "Variable-length array type capture cannot have a variable!");
    break;
  }
}

CapturedStmt::VariableCaptureKind
CapturedStmt::Capture::getCaptureKind() const {
  return VarAndKind.getInt();
}

VarDecl *CapturedStmt::Capture::getCapturedVar() const {
  assert((capturesVariable() || capturesVariableByCopy()) &&
         "No variable available for 'this' or VAT capture");
  return VarAndKind.getPointer();
}

CapturedStmt::Capture *CapturedStmt::getStoredCaptures() const {
  unsigned Size = sizeof(CapturedStmt) + sizeof(Stmt *) * (NumCaptures + 1);

  // Offset of the first Capture object.
  unsigned FirstCaptureOffset = llvm::alignTo(Size, alignof(Capture));

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
    Size = llvm::alignTo(Size, alignof(Capture));
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
    Size = llvm::alignTo(Size, alignof(Capture));
    Size += sizeof(Capture) * NumCaptures;
  }

  void *Mem = Context.Allocate(Size);
  return new (Mem) CapturedStmt(EmptyShell(), NumCaptures);
}

Stmt::child_range CapturedStmt::children() {
  // Children are captured field initilizers.
  return child_range(getStoredStmts(), getStoredStmts() + NumCaptures);
}

CapturedDecl *CapturedStmt::getCapturedDecl() {
  return CapDeclAndKind.getPointer();
}
const CapturedDecl *CapturedStmt::getCapturedDecl() const {
  return CapDeclAndKind.getPointer();
}

/// \brief Set the outlined function declaration.
void CapturedStmt::setCapturedDecl(CapturedDecl *D) {
  assert(D && "null CapturedDecl");
  CapDeclAndKind.setPointer(D);
}

/// \brief Retrieve the captured region kind.
CapturedRegionKind CapturedStmt::getCapturedRegionKind() const {
  return CapDeclAndKind.getInt();
}

/// \brief Set the captured region kind.
void CapturedStmt::setCapturedRegionKind(CapturedRegionKind Kind) {
  CapDeclAndKind.setInt(Kind);
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
