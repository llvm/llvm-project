//===--- ExprConstant.cpp - Expression Constant Evaluator -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expr constant evaluator.
//
// Constant expression evaluation produces four main results:
//
//  * A success/failure flag indicating whether constant folding was successful.
//    This is the 'bool' return value used by most of the code in this file. A
//    'false' return value indicates that constant folding has failed, and any
//    appropriate diagnostic has already been produced.
//
//  * An evaluated result, valid only if constant folding has not failed.
//
//  * A flag indicating if evaluation encountered (unevaluated) side-effects.
//    These arise in cases such as (sideEffect(), 0) and (sideEffect() || 1),
//    where it is possible to determine the evaluated result regardless.
//
//  * A set of notes indicating why the evaluation was not a constant expression
//    (under the C++11 rules only, at the moment), or, if folding failed too,
//    why the expression could not be folded.
//
// If we are checking for a potential constant expression, failure to constant
// fold a potential constant sub-expression will be indicated by a 'false'
// return value (the expression could not be folded) and no diagnostic (the
// expression is not necessarily non-constant).
//
//===----------------------------------------------------------------------===//

#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include <cstring>
#include <functional>

using namespace clang;
using llvm::APSInt;
using llvm::APFloat;

static bool IsGlobalLValue(APValue::LValueBase B);

namespace {
  struct LValue;
  struct CallStackFrame;
  struct EvalInfo;

  static QualType getType(APValue::LValueBase B) {
    if (!B) return QualType();
    if (const ValueDecl *D = B.dyn_cast<const ValueDecl*>())
      return D->getType();
    return B.get<const Expr*>()->getType();
  }

  /// Get an LValue path entry, which is known to not be an array index, as a
  /// field or base class.
  static
  APValue::BaseOrMemberType getAsBaseOrMember(APValue::LValuePathEntry E) {
    APValue::BaseOrMemberType Value;
    Value.setFromOpaqueValue(E.BaseOrMember);
    return Value;
  }

  /// Get an LValue path entry, which is known to not be an array index, as a
  /// field declaration.
  static const FieldDecl *getAsField(APValue::LValuePathEntry E) {
    return dyn_cast<FieldDecl>(getAsBaseOrMember(E).getPointer());
  }
  /// Get an LValue path entry, which is known to not be an array index, as a
  /// base class declaration.
  static const CXXRecordDecl *getAsBaseClass(APValue::LValuePathEntry E) {
    return dyn_cast<CXXRecordDecl>(getAsBaseOrMember(E).getPointer());
  }
  /// Determine whether this LValue path entry for a base class names a virtual
  /// base class.
  static bool isVirtualBaseClass(APValue::LValuePathEntry E) {
    return getAsBaseOrMember(E).getInt();
  }

  /// Find the path length and type of the most-derived subobject in the given
  /// path, and find the size of the containing array, if any.
  static
  unsigned findMostDerivedSubobject(ASTContext &Ctx, QualType Base,
                                    ArrayRef<APValue::LValuePathEntry> Path,
                                    uint64_t &ArraySize, QualType &Type) {
    unsigned MostDerivedLength = 0;
    Type = Base;
    for (unsigned I = 0, N = Path.size(); I != N; ++I) {
      if (Type->isArrayType()) {
        const ConstantArrayType *CAT =
          cast<ConstantArrayType>(Ctx.getAsArrayType(Type));
        Type = CAT->getElementType();
        ArraySize = CAT->getSize().getZExtValue();
        MostDerivedLength = I + 1;
      } else if (Type->isAnyComplexType()) {
        const ComplexType *CT = Type->castAs<ComplexType>();
        Type = CT->getElementType();
        ArraySize = 2;
        MostDerivedLength = I + 1;
      } else if (const FieldDecl *FD = getAsField(Path[I])) {
        Type = FD->getType();
        ArraySize = 0;
        MostDerivedLength = I + 1;
      } else {
        // Path[I] describes a base class.
        ArraySize = 0;
      }
    }
    return MostDerivedLength;
  }

  // The order of this enum is important for diagnostics.
  enum CheckSubobjectKind {
    CSK_Base, CSK_Derived, CSK_Field, CSK_ArrayToPointer, CSK_ArrayIndex,
    CSK_This, CSK_Real, CSK_Imag
  };

  /// A path from a glvalue to a subobject of that glvalue.
  struct SubobjectDesignator {
    /// True if the subobject was named in a manner not supported by C++11. Such
    /// lvalues can still be folded, but they are not core constant expressions
    /// and we cannot perform lvalue-to-rvalue conversions on them.
    bool Invalid : 1;

    /// Is this a pointer one past the end of an object?
    bool IsOnePastTheEnd : 1;

    /// The length of the path to the most-derived object of which this is a
    /// subobject.
    unsigned MostDerivedPathLength : 30;

    /// The size of the array of which the most-derived object is an element, or
    /// 0 if the most-derived object is not an array element.
    uint64_t MostDerivedArraySize;

    /// The type of the most derived object referred to by this address.
    QualType MostDerivedType;

    typedef APValue::LValuePathEntry PathEntry;

    /// The entries on the path from the glvalue to the designated subobject.
    SmallVector<PathEntry, 8> Entries;

    SubobjectDesignator() : Invalid(true) {}

    explicit SubobjectDesignator(QualType T)
      : Invalid(false), IsOnePastTheEnd(false), MostDerivedPathLength(0),
        MostDerivedArraySize(0), MostDerivedType(T) {}

    SubobjectDesignator(ASTContext &Ctx, const APValue &V)
      : Invalid(!V.isLValue() || !V.hasLValuePath()), IsOnePastTheEnd(false),
        MostDerivedPathLength(0), MostDerivedArraySize(0) {
      if (!Invalid) {
        IsOnePastTheEnd = V.isLValueOnePastTheEnd();
        ArrayRef<PathEntry> VEntries = V.getLValuePath();
        Entries.insert(Entries.end(), VEntries.begin(), VEntries.end());
        if (V.getLValueBase())
          MostDerivedPathLength =
              findMostDerivedSubobject(Ctx, getType(V.getLValueBase()),
                                       V.getLValuePath(), MostDerivedArraySize,
                                       MostDerivedType);
      }
    }

    void setInvalid() {
      Invalid = true;
      Entries.clear();
    }

    /// Determine whether this is a one-past-the-end pointer.
    bool isOnePastTheEnd() const {
      if (IsOnePastTheEnd)
        return true;
      if (MostDerivedArraySize &&
          Entries[MostDerivedPathLength - 1].ArrayIndex == MostDerivedArraySize)
        return true;
      return false;
    }

    /// Check that this refers to a valid subobject.
    bool isValidSubobject() const {
      if (Invalid)
        return false;
      return !isOnePastTheEnd();
    }
    /// Check that this refers to a valid subobject, and if not, produce a
    /// relevant diagnostic and set the designator as invalid.
    bool checkSubobject(EvalInfo &Info, const Expr *E, CheckSubobjectKind CSK);

    /// Update this designator to refer to the first element within this array.
    void addArrayUnchecked(const ConstantArrayType *CAT) {
      PathEntry Entry;
      Entry.ArrayIndex = 0;
      Entries.push_back(Entry);

      // This is a most-derived object.
      MostDerivedType = CAT->getElementType();
      MostDerivedArraySize = CAT->getSize().getZExtValue();
      MostDerivedPathLength = Entries.size();
    }
    /// Update this designator to refer to the given base or member of this
    /// object.
    void addDeclUnchecked(const Decl *D, bool Virtual = false) {
      PathEntry Entry;
      APValue::BaseOrMemberType Value(D, Virtual);
      Entry.BaseOrMember = Value.getOpaqueValue();
      Entries.push_back(Entry);

      // If this isn't a base class, it's a new most-derived object.
      if (const FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
        MostDerivedType = FD->getType();
        MostDerivedArraySize = 0;
        MostDerivedPathLength = Entries.size();
      }
    }
    /// Update this designator to refer to the given complex component.
    void addComplexUnchecked(QualType EltTy, bool Imag) {
      PathEntry Entry;
      Entry.ArrayIndex = Imag;
      Entries.push_back(Entry);

      // This is technically a most-derived object, though in practice this
      // is unlikely to matter.
      MostDerivedType = EltTy;
      MostDerivedArraySize = 2;
      MostDerivedPathLength = Entries.size();
    }
    void diagnosePointerArithmetic(EvalInfo &Info, const Expr *E, uint64_t N);
    /// Add N to the address of this subobject.
    void adjustIndex(EvalInfo &Info, const Expr *E, uint64_t N) {
      if (Invalid) return;
      if (MostDerivedPathLength == Entries.size() && MostDerivedArraySize) {
        Entries.back().ArrayIndex += N;
        if (Entries.back().ArrayIndex > MostDerivedArraySize) {
          diagnosePointerArithmetic(Info, E, Entries.back().ArrayIndex);
          setInvalid();
        }
        return;
      }
      // [expr.add]p4: For the purposes of these operators, a pointer to a
      // nonarray object behaves the same as a pointer to the first element of
      // an array of length one with the type of the object as its element type.
      if (IsOnePastTheEnd && N == (uint64_t)-1)
        IsOnePastTheEnd = false;
      else if (!IsOnePastTheEnd && N == 1)
        IsOnePastTheEnd = true;
      else if (N != 0) {
        diagnosePointerArithmetic(Info, E, uint64_t(IsOnePastTheEnd) + N);
        setInvalid();
      }
    }
  };

  /// A stack frame in the constexpr call stack.
  struct CallStackFrame {
    EvalInfo &Info;

    /// Parent - The caller of this stack frame.
    CallStackFrame *Caller;

    /// CallLoc - The location of the call expression for this call.
    SourceLocation CallLoc;

    /// Callee - The function which was called.
    const FunctionDecl *Callee;

    /// Index - The call index of this call.
    unsigned Index;

    /// This - The binding for the this pointer in this call, if any.
    const LValue *This;

    /// ParmBindings - Parameter bindings for this function call, indexed by
    /// parameters' function scope indices.
    const APValue *Arguments;

    // Note that we intentionally use std::map here so that references to
    // values are stable.
    typedef std::map<const Expr*, APValue> MapTy;
    typedef MapTy::const_iterator temp_iterator;
    /// Temporaries - Temporary lvalues materialized within this stack frame.
    MapTy Temporaries;

    CallStackFrame(EvalInfo &Info, SourceLocation CallLoc,
                   const FunctionDecl *Callee, const LValue *This,
                   const APValue *Arguments);
    ~CallStackFrame();
  };

  /// A partial diagnostic which we might know in advance that we are not going
  /// to emit.
  class OptionalDiagnostic {
    PartialDiagnostic *Diag;

  public:
    explicit OptionalDiagnostic(PartialDiagnostic *Diag = 0) : Diag(Diag) {}

    template<typename T>
    OptionalDiagnostic &operator<<(const T &v) {
      if (Diag)
        *Diag << v;
      return *this;
    }

    OptionalDiagnostic &operator<<(const APSInt &I) {
      if (Diag) {
        llvm::SmallVector<char, 32> Buffer;
        I.toString(Buffer);
        *Diag << StringRef(Buffer.data(), Buffer.size());
      }
      return *this;
    }

    OptionalDiagnostic &operator<<(const APFloat &F) {
      if (Diag) {
        llvm::SmallVector<char, 32> Buffer;
        F.toString(Buffer);
        *Diag << StringRef(Buffer.data(), Buffer.size());
      }
      return *this;
    }
  };

  /// EvalInfo - This is a private struct used by the evaluator to capture
  /// information about a subexpression as it is folded.  It retains information
  /// about the AST context, but also maintains information about the folded
  /// expression.
  ///
  /// If an expression could be evaluated, it is still possible it is not a C
  /// "integer constant expression" or constant expression.  If not, this struct
  /// captures information about how and why not.
  ///
  /// One bit of information passed *into* the request for constant folding
  /// indicates whether the subexpression is "evaluated" or not according to C
  /// rules.  For example, the RHS of (0 && foo()) is not evaluated.  We can
  /// evaluate the expression regardless of what the RHS is, but C only allows
  /// certain things in certain situations.
  struct EvalInfo {
    ASTContext &Ctx;

    /// EvalStatus - Contains information about the evaluation.
    Expr::EvalStatus &EvalStatus;

    /// CurrentCall - The top of the constexpr call stack.
    CallStackFrame *CurrentCall;

    /// CallStackDepth - The number of calls in the call stack right now.
    unsigned CallStackDepth;

    /// NextCallIndex - The next call index to assign.
    unsigned NextCallIndex;

    /// BottomFrame - The frame in which evaluation started. This must be
    /// initialized after CurrentCall and CallStackDepth.
    CallStackFrame BottomFrame;

    /// EvaluatingDecl - This is the declaration whose initializer is being
    /// evaluated, if any.
    const VarDecl *EvaluatingDecl;

    /// EvaluatingDeclValue - This is the value being constructed for the
    /// declaration whose initializer is being evaluated, if any.
    APValue *EvaluatingDeclValue;

    /// HasActiveDiagnostic - Was the previous diagnostic stored? If so, further
    /// notes attached to it will also be stored, otherwise they will not be.
    bool HasActiveDiagnostic;

    /// CheckingPotentialConstantExpression - Are we checking whether the
    /// expression is a potential constant expression? If so, some diagnostics
    /// are suppressed.
    bool CheckingPotentialConstantExpression;

    EvalInfo(const ASTContext &C, Expr::EvalStatus &S)
      : Ctx(const_cast<ASTContext&>(C)), EvalStatus(S), CurrentCall(0),
        CallStackDepth(0), NextCallIndex(1),
        BottomFrame(*this, SourceLocation(), 0, 0, 0),
        EvaluatingDecl(0), EvaluatingDeclValue(0), HasActiveDiagnostic(false),
        CheckingPotentialConstantExpression(false) {}

    void setEvaluatingDecl(const VarDecl *VD, APValue &Value) {
      EvaluatingDecl = VD;
      EvaluatingDeclValue = &Value;
    }

    const LangOptions &getLangOpts() const { return Ctx.getLangOpts(); }

    bool CheckCallLimit(SourceLocation Loc) {
      // Don't perform any constexpr calls (other than the call we're checking)
      // when checking a potential constant expression.
      if (CheckingPotentialConstantExpression && CallStackDepth > 1)
        return false;
      if (NextCallIndex == 0) {
        // NextCallIndex has wrapped around.
        Diag(Loc, diag::note_constexpr_call_limit_exceeded);
        return false;
      }
      if (CallStackDepth <= getLangOpts().ConstexprCallDepth)
        return true;
      Diag(Loc, diag::note_constexpr_depth_limit_exceeded)
        << getLangOpts().ConstexprCallDepth;
      return false;
    }

    CallStackFrame *getCallFrame(unsigned CallIndex) {
      assert(CallIndex && "no call index in getCallFrame");
      // We will eventually hit BottomFrame, which has Index 1, so Frame can't
      // be null in this loop.
      CallStackFrame *Frame = CurrentCall;
      while (Frame->Index > CallIndex)
        Frame = Frame->Caller;
      return (Frame->Index == CallIndex) ? Frame : 0;
    }

  private:
    /// Add a diagnostic to the diagnostics list.
    PartialDiagnostic &addDiag(SourceLocation Loc, diag::kind DiagId) {
      PartialDiagnostic PD(DiagId, Ctx.getDiagAllocator());
      EvalStatus.Diag->push_back(std::make_pair(Loc, PD));
      return EvalStatus.Diag->back().second;
    }

    /// Add notes containing a call stack to the current point of evaluation.
    void addCallStack(unsigned Limit);

  public:
    /// Diagnose that the evaluation cannot be folded.
    OptionalDiagnostic Diag(SourceLocation Loc, diag::kind DiagId
                              = diag::note_invalid_subexpr_in_const_expr,
                            unsigned ExtraNotes = 0) {
      // If we have a prior diagnostic, it will be noting that the expression
      // isn't a constant expression. This diagnostic is more important.
      // FIXME: We might want to show both diagnostics to the user.
      if (EvalStatus.Diag) {
        unsigned CallStackNotes = CallStackDepth - 1;
        unsigned Limit = Ctx.getDiagnostics().getConstexprBacktraceLimit();
        if (Limit)
          CallStackNotes = std::min(CallStackNotes, Limit + 1);
        if (CheckingPotentialConstantExpression)
          CallStackNotes = 0;

        HasActiveDiagnostic = true;
        EvalStatus.Diag->clear();
        EvalStatus.Diag->reserve(1 + ExtraNotes + CallStackNotes);
        addDiag(Loc, DiagId);
        if (!CheckingPotentialConstantExpression)
          addCallStack(Limit);
        return OptionalDiagnostic(&(*EvalStatus.Diag)[0].second);
      }
      HasActiveDiagnostic = false;
      return OptionalDiagnostic();
    }

    OptionalDiagnostic Diag(const Expr *E, diag::kind DiagId
                              = diag::note_invalid_subexpr_in_const_expr,
                            unsigned ExtraNotes = 0) {
      if (EvalStatus.Diag)
        return Diag(E->getExprLoc(), DiagId, ExtraNotes);
      HasActiveDiagnostic = false;
      return OptionalDiagnostic();
    }

    /// Diagnose that the evaluation does not produce a C++11 core constant
    /// expression.
    template<typename LocArg>
    OptionalDiagnostic CCEDiag(LocArg Loc, diag::kind DiagId
                                 = diag::note_invalid_subexpr_in_const_expr,
                               unsigned ExtraNotes = 0) {
      // Don't override a previous diagnostic.
      if (!EvalStatus.Diag || !EvalStatus.Diag->empty()) {
        HasActiveDiagnostic = false;
        return OptionalDiagnostic();
      }
      return Diag(Loc, DiagId, ExtraNotes);
    }

    /// Add a note to a prior diagnostic.
    OptionalDiagnostic Note(SourceLocation Loc, diag::kind DiagId) {
      if (!HasActiveDiagnostic)
        return OptionalDiagnostic();
      return OptionalDiagnostic(&addDiag(Loc, DiagId));
    }

    /// Add a stack of notes to a prior diagnostic.
    void addNotes(ArrayRef<PartialDiagnosticAt> Diags) {
      if (HasActiveDiagnostic) {
        EvalStatus.Diag->insert(EvalStatus.Diag->end(),
                                Diags.begin(), Diags.end());
      }
    }

    /// Should we continue evaluation as much as possible after encountering a
    /// construct which can't be folded?
    bool keepEvaluatingAfterFailure() {
      return CheckingPotentialConstantExpression &&
             EvalStatus.Diag && EvalStatus.Diag->empty();
    }
  };

  /// Object used to treat all foldable expressions as constant expressions.
  struct FoldConstant {
    bool Enabled;

    explicit FoldConstant(EvalInfo &Info)
      : Enabled(Info.EvalStatus.Diag && Info.EvalStatus.Diag->empty() &&
                !Info.EvalStatus.HasSideEffects) {
    }
    // Treat the value we've computed since this object was created as constant.
    void Fold(EvalInfo &Info) {
      if (Enabled && !Info.EvalStatus.Diag->empty() &&
          !Info.EvalStatus.HasSideEffects)
        Info.EvalStatus.Diag->clear();
    }
  };

  /// RAII object used to suppress diagnostics and side-effects from a
  /// speculative evaluation.
  class SpeculativeEvaluationRAII {
    EvalInfo &Info;
    Expr::EvalStatus Old;

  public:
    SpeculativeEvaluationRAII(EvalInfo &Info,
                              llvm::SmallVectorImpl<PartialDiagnosticAt>
                                *NewDiag = 0)
      : Info(Info), Old(Info.EvalStatus) {
      Info.EvalStatus.Diag = NewDiag;
    }
    ~SpeculativeEvaluationRAII() {
      Info.EvalStatus = Old;
    }
  };
}

bool SubobjectDesignator::checkSubobject(EvalInfo &Info, const Expr *E,
                                         CheckSubobjectKind CSK) {
  if (Invalid)
    return false;
  if (isOnePastTheEnd()) {
    Info.CCEDiag(E, diag::note_constexpr_past_end_subobject)
      << CSK;
    setInvalid();
    return false;
  }
  return true;
}

void SubobjectDesignator::diagnosePointerArithmetic(EvalInfo &Info,
                                                    const Expr *E, uint64_t N) {
  if (MostDerivedPathLength == Entries.size() && MostDerivedArraySize)
    Info.CCEDiag(E, diag::note_constexpr_array_index)
      << static_cast<int>(N) << /*array*/ 0
      << static_cast<unsigned>(MostDerivedArraySize);
  else
    Info.CCEDiag(E, diag::note_constexpr_array_index)
      << static_cast<int>(N) << /*non-array*/ 1;
  setInvalid();
}

CallStackFrame::CallStackFrame(EvalInfo &Info, SourceLocation CallLoc,
                               const FunctionDecl *Callee, const LValue *This,
                               const APValue *Arguments)
    : Info(Info), Caller(Info.CurrentCall), CallLoc(CallLoc), Callee(Callee),
      Index(Info.NextCallIndex++), This(This), Arguments(Arguments) {
  Info.CurrentCall = this;
  ++Info.CallStackDepth;
}

CallStackFrame::~CallStackFrame() {
  assert(Info.CurrentCall == this && "calls retired out of order");
  --Info.CallStackDepth;
  Info.CurrentCall = Caller;
}

/// Produce a string describing the given constexpr call.
static void describeCall(CallStackFrame *Frame, llvm::raw_ostream &Out) {
  unsigned ArgIndex = 0;
  bool IsMemberCall = isa<CXXMethodDecl>(Frame->Callee) &&
                      !isa<CXXConstructorDecl>(Frame->Callee) &&
                      cast<CXXMethodDecl>(Frame->Callee)->isInstance();

  if (!IsMemberCall)
    Out << *Frame->Callee << '(';

  for (FunctionDecl::param_const_iterator I = Frame->Callee->param_begin(),
       E = Frame->Callee->param_end(); I != E; ++I, ++ArgIndex) {
    if (ArgIndex > (unsigned)IsMemberCall)
      Out << ", ";

    const ParmVarDecl *Param = *I;
    const APValue &Arg = Frame->Arguments[ArgIndex];
    Arg.printPretty(Out, Frame->Info.Ctx, Param->getType());

    if (ArgIndex == 0 && IsMemberCall)
      Out << "->" << *Frame->Callee << '(';
  }

  Out << ')';
}

void EvalInfo::addCallStack(unsigned Limit) {
  // Determine which calls to skip, if any.
  unsigned ActiveCalls = CallStackDepth - 1;
  unsigned SkipStart = ActiveCalls, SkipEnd = SkipStart;
  if (Limit && Limit < ActiveCalls) {
    SkipStart = Limit / 2 + Limit % 2;
    SkipEnd = ActiveCalls - Limit / 2;
  }

  // Walk the call stack and add the diagnostics.
  unsigned CallIdx = 0;
  for (CallStackFrame *Frame = CurrentCall; Frame != &BottomFrame;
       Frame = Frame->Caller, ++CallIdx) {
    // Skip this call?
    if (CallIdx >= SkipStart && CallIdx < SkipEnd) {
      if (CallIdx == SkipStart) {
        // Note that we're skipping calls.
        addDiag(Frame->CallLoc, diag::note_constexpr_calls_suppressed)
          << unsigned(ActiveCalls - Limit);
      }
      continue;
    }

    llvm::SmallVector<char, 128> Buffer;
    llvm::raw_svector_ostream Out(Buffer);
    describeCall(Frame, Out);
    addDiag(Frame->CallLoc, diag::note_constexpr_call_here) << Out.str();
  }
}

namespace {
  struct ComplexValue {
  private:
    bool IsInt;

  public:
    APSInt IntReal, IntImag;
    APFloat FloatReal, FloatImag;

    ComplexValue() : FloatReal(APFloat::Bogus), FloatImag(APFloat::Bogus) {}

    void makeComplexFloat() { IsInt = false; }
    bool isComplexFloat() const { return !IsInt; }
    APFloat &getComplexFloatReal() { return FloatReal; }
    APFloat &getComplexFloatImag() { return FloatImag; }

    void makeComplexInt() { IsInt = true; }
    bool isComplexInt() const { return IsInt; }
    APSInt &getComplexIntReal() { return IntReal; }
    APSInt &getComplexIntImag() { return IntImag; }

    void moveInto(APValue &v) const {
      if (isComplexFloat())
        v = APValue(FloatReal, FloatImag);
      else
        v = APValue(IntReal, IntImag);
    }
    void setFrom(const APValue &v) {
      assert(v.isComplexFloat() || v.isComplexInt());
      if (v.isComplexFloat()) {
        makeComplexFloat();
        FloatReal = v.getComplexFloatReal();
        FloatImag = v.getComplexFloatImag();
      } else {
        makeComplexInt();
        IntReal = v.getComplexIntReal();
        IntImag = v.getComplexIntImag();
      }
    }
  };

  struct LValue {
    APValue::LValueBase Base;
    CharUnits Offset;
    unsigned CallIndex;
    SubobjectDesignator Designator;

    const APValue::LValueBase getLValueBase() const { return Base; }
    CharUnits &getLValueOffset() { return Offset; }
    const CharUnits &getLValueOffset() const { return Offset; }
    unsigned getLValueCallIndex() const { return CallIndex; }
    SubobjectDesignator &getLValueDesignator() { return Designator; }
    const SubobjectDesignator &getLValueDesignator() const { return Designator;}

    void moveInto(APValue &V) const {
      if (Designator.Invalid)
        V = APValue(Base, Offset, APValue::NoLValuePath(), CallIndex);
      else
        V = APValue(Base, Offset, Designator.Entries,
                    Designator.IsOnePastTheEnd, CallIndex);
    }
    void setFrom(ASTContext &Ctx, const APValue &V) {
      assert(V.isLValue());
      Base = V.getLValueBase();
      Offset = V.getLValueOffset();
      CallIndex = V.getLValueCallIndex();
      Designator = SubobjectDesignator(Ctx, V);
    }

    void set(APValue::LValueBase B, unsigned I = 0) {
      Base = B;
      Offset = CharUnits::Zero();
      CallIndex = I;
      Designator = SubobjectDesignator(getType(B));
    }

    // Check that this LValue is not based on a null pointer. If it is, produce
    // a diagnostic and mark the designator as invalid.
    bool checkNullPointer(EvalInfo &Info, const Expr *E,
                          CheckSubobjectKind CSK) {
      if (Designator.Invalid)
        return false;
      if (!Base) {
        Info.CCEDiag(E, diag::note_constexpr_null_subobject)
          << CSK;
        Designator.setInvalid();
        return false;
      }
      return true;
    }

    // Check this LValue refers to an object. If not, set the designator to be
    // invalid and emit a diagnostic.
    bool checkSubobject(EvalInfo &Info, const Expr *E, CheckSubobjectKind CSK) {
      // Outside C++11, do not build a designator referring to a subobject of
      // any object: we won't use such a designator for anything.
      if (!Info.getLangOpts().CPlusPlus0x)
        Designator.setInvalid();
      return checkNullPointer(Info, E, CSK) &&
             Designator.checkSubobject(Info, E, CSK);
    }

    void addDecl(EvalInfo &Info, const Expr *E,
                 const Decl *D, bool Virtual = false) {
      if (checkSubobject(Info, E, isa<FieldDecl>(D) ? CSK_Field : CSK_Base))
        Designator.addDeclUnchecked(D, Virtual);
    }
    void addArray(EvalInfo &Info, const Expr *E, const ConstantArrayType *CAT) {
      if (checkSubobject(Info, E, CSK_ArrayToPointer))
        Designator.addArrayUnchecked(CAT);
    }
    void addComplex(EvalInfo &Info, const Expr *E, QualType EltTy, bool Imag) {
      if (checkSubobject(Info, E, Imag ? CSK_Imag : CSK_Real))
        Designator.addComplexUnchecked(EltTy, Imag);
    }
    void adjustIndex(EvalInfo &Info, const Expr *E, uint64_t N) {
      if (checkNullPointer(Info, E, CSK_ArrayIndex))
        Designator.adjustIndex(Info, E, N);
    }
  };

  struct MemberPtr {
    MemberPtr() {}
    explicit MemberPtr(const ValueDecl *Decl) :
      DeclAndIsDerivedMember(Decl, false), Path() {}

    /// The member or (direct or indirect) field referred to by this member
    /// pointer, or 0 if this is a null member pointer.
    const ValueDecl *getDecl() const {
      return DeclAndIsDerivedMember.getPointer();
    }
    /// Is this actually a member of some type derived from the relevant class?
    bool isDerivedMember() const {
      return DeclAndIsDerivedMember.getInt();
    }
    /// Get the class which the declaration actually lives in.
    const CXXRecordDecl *getContainingRecord() const {
      return cast<CXXRecordDecl>(
          DeclAndIsDerivedMember.getPointer()->getDeclContext());
    }

    void moveInto(APValue &V) const {
      V = APValue(getDecl(), isDerivedMember(), Path);
    }
    void setFrom(const APValue &V) {
      assert(V.isMemberPointer());
      DeclAndIsDerivedMember.setPointer(V.getMemberPointerDecl());
      DeclAndIsDerivedMember.setInt(V.isMemberPointerToDerivedMember());
      Path.clear();
      ArrayRef<const CXXRecordDecl*> P = V.getMemberPointerPath();
      Path.insert(Path.end(), P.begin(), P.end());
    }

    /// DeclAndIsDerivedMember - The member declaration, and a flag indicating
    /// whether the member is a member of some class derived from the class type
    /// of the member pointer.
    llvm::PointerIntPair<const ValueDecl*, 1, bool> DeclAndIsDerivedMember;
    /// Path - The path of base/derived classes from the member declaration's
    /// class (exclusive) to the class type of the member pointer (inclusive).
    SmallVector<const CXXRecordDecl*, 4> Path;

    /// Perform a cast towards the class of the Decl (either up or down the
    /// hierarchy).
    bool castBack(const CXXRecordDecl *Class) {
      assert(!Path.empty());
      const CXXRecordDecl *Expected;
      if (Path.size() >= 2)
        Expected = Path[Path.size() - 2];
      else
        Expected = getContainingRecord();
      if (Expected->getCanonicalDecl() != Class->getCanonicalDecl()) {
        // C++11 [expr.static.cast]p12: In a conversion from (D::*) to (B::*),
        // if B does not contain the original member and is not a base or
        // derived class of the class containing the original member, the result
        // of the cast is undefined.
        // C++11 [conv.mem]p2 does not cover this case for a cast from (B::*) to
        // (D::*). We consider that to be a language defect.
        return false;
      }
      Path.pop_back();
      return true;
    }
    /// Perform a base-to-derived member pointer cast.
    bool castToDerived(const CXXRecordDecl *Derived) {
      if (!getDecl())
        return true;
      if (!isDerivedMember()) {
        Path.push_back(Derived);
        return true;
      }
      if (!castBack(Derived))
        return false;
      if (Path.empty())
        DeclAndIsDerivedMember.setInt(false);
      return true;
    }
    /// Perform a derived-to-base member pointer cast.
    bool castToBase(const CXXRecordDecl *Base) {
      if (!getDecl())
        return true;
      if (Path.empty())
        DeclAndIsDerivedMember.setInt(true);
      if (isDerivedMember()) {
        Path.push_back(Base);
        return true;
      }
      return castBack(Base);
    }
  };

  /// Compare two member pointers, which are assumed to be of the same type.
  static bool operator==(const MemberPtr &LHS, const MemberPtr &RHS) {
    if (!LHS.getDecl() || !RHS.getDecl())
      return !LHS.getDecl() && !RHS.getDecl();
    if (LHS.getDecl()->getCanonicalDecl() != RHS.getDecl()->getCanonicalDecl())
      return false;
    return LHS.Path == RHS.Path;
  }

  /// Kinds of constant expression checking, for diagnostics.
  enum CheckConstantExpressionKind {
    CCEK_Constant,    ///< A normal constant.
    CCEK_ReturnValue, ///< A constexpr function return value.
    CCEK_MemberInit   ///< A constexpr constructor mem-initializer.
  };
}

static bool Evaluate(APValue &Result, EvalInfo &Info, const Expr *E);
static bool EvaluateInPlace(APValue &Result, EvalInfo &Info,
                            const LValue &This, const Expr *E,
                            CheckConstantExpressionKind CCEK = CCEK_Constant,
                            bool AllowNonLiteralTypes = false);
static bool EvaluateLValue(const Expr *E, LValue &Result, EvalInfo &Info);
static bool EvaluatePointer(const Expr *E, LValue &Result, EvalInfo &Info);
static bool EvaluateMemberPointer(const Expr *E, MemberPtr &Result,
                                  EvalInfo &Info);
static bool EvaluateTemporary(const Expr *E, LValue &Result, EvalInfo &Info);
static bool EvaluateInteger(const Expr *E, APSInt  &Result, EvalInfo &Info);
static bool EvaluateIntegerOrLValue(const Expr *E, APValue &Result,
                                    EvalInfo &Info);
static bool EvaluateFloat(const Expr *E, APFloat &Result, EvalInfo &Info);
static bool EvaluateComplex(const Expr *E, ComplexValue &Res, EvalInfo &Info);

//===----------------------------------------------------------------------===//
// Misc utilities
//===----------------------------------------------------------------------===//

/// Should this call expression be treated as a string literal?
static bool IsStringLiteralCall(const CallExpr *E) {
  unsigned Builtin = E->isBuiltinCall();
  return (Builtin == Builtin::BI__builtin___CFStringMakeConstantString ||
          Builtin == Builtin::BI__builtin___NSStringMakeConstantString);
}

static bool IsGlobalLValue(APValue::LValueBase B) {
  // C++11 [expr.const]p3 An address constant expression is a prvalue core
  // constant expression of pointer type that evaluates to...

  // ... a null pointer value, or a prvalue core constant expression of type
  // std::nullptr_t.
  if (!B) return true;

  if (const ValueDecl *D = B.dyn_cast<const ValueDecl*>()) {
    // ... the address of an object with static storage duration,
    if (const VarDecl *VD = dyn_cast<VarDecl>(D))
      return VD->hasGlobalStorage();
    // ... the address of a function,
    return isa<FunctionDecl>(D);
  }

  const Expr *E = B.get<const Expr*>();
  switch (E->getStmtClass()) {
  default:
    return false;
  case Expr::CompoundLiteralExprClass: {
    const CompoundLiteralExpr *CLE = cast<CompoundLiteralExpr>(E);
    return CLE->isFileScope() && CLE->isLValue();
  }
  // A string literal has static storage duration.
  case Expr::StringLiteralClass:
  case Expr::PredefinedExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::ObjCEncodeExprClass:
  case Expr::CXXTypeidExprClass:
  case Expr::CXXUuidofExprClass:
    return true;
  case Expr::CallExprClass:
    return IsStringLiteralCall(cast<CallExpr>(E));
  // For GCC compatibility, &&label has static storage duration.
  case Expr::AddrLabelExprClass:
    return true;
  // A Block literal expression may be used as the initialization value for
  // Block variables at global or local static scope.
  case Expr::BlockExprClass:
    return !cast<BlockExpr>(E)->getBlockDecl()->hasCaptures();
  case Expr::ImplicitValueInitExprClass:
    // FIXME:
    // We can never form an lvalue with an implicit value initialization as its
    // base through expression evaluation, so these only appear in one case: the
    // implicit variable declaration we invent when checking whether a constexpr
    // constructor can produce a constant expression. We must assume that such
    // an expression might be a global lvalue.
    return true;
  }
}

static void NoteLValueLocation(EvalInfo &Info, APValue::LValueBase Base) {
  assert(Base && "no location for a null lvalue");
  const ValueDecl *VD = Base.dyn_cast<const ValueDecl*>();
  if (VD)
    Info.Note(VD->getLocation(), diag::note_declared_at);
  else
    Info.Note(Base.get<const Expr*>()->getExprLoc(),
              diag::note_constexpr_temporary_here);
}

/// Check that this reference or pointer core constant expression is a valid
/// value for an address or reference constant expression. Return true if we
/// can fold this expression, whether or not it's a constant expression.
static bool CheckLValueConstantExpression(EvalInfo &Info, SourceLocation Loc,
                                          QualType Type, const LValue &LVal) {
  bool IsReferenceType = Type->isReferenceType();

  APValue::LValueBase Base = LVal.getLValueBase();
  const SubobjectDesignator &Designator = LVal.getLValueDesignator();

  // Check that the object is a global. Note that the fake 'this' object we
  // manufacture when checking potential constant expressions is conservatively
  // assumed to be global here.
  if (!IsGlobalLValue(Base)) {
    if (Info.getLangOpts().CPlusPlus0x) {
      const ValueDecl *VD = Base.dyn_cast<const ValueDecl*>();
      Info.Diag(Loc, diag::note_constexpr_non_global, 1)
        << IsReferenceType << !Designator.Entries.empty()
        << !!VD << VD;
      NoteLValueLocation(Info, Base);
    } else {
      Info.Diag(Loc);
    }
    // Don't allow references to temporaries to escape.
    return false;
  }
  assert((Info.CheckingPotentialConstantExpression ||
          LVal.getLValueCallIndex() == 0) &&
         "have call index for global lvalue");

  // Check if this is a thread-local variable.
  if (const ValueDecl *VD = Base.dyn_cast<const ValueDecl*>()) {
    if (const VarDecl *Var = dyn_cast<const VarDecl>(VD)) {
      if (Var->isThreadSpecified())
        return false;
    }
  }

  // Allow address constant expressions to be past-the-end pointers. This is
  // an extension: the standard requires them to point to an object.
  if (!IsReferenceType)
    return true;

  // A reference constant expression must refer to an object.
  if (!Base) {
    // FIXME: diagnostic
    Info.CCEDiag(Loc);
    return true;
  }

  // Does this refer one past the end of some object?
  if (Designator.isOnePastTheEnd()) {
    const ValueDecl *VD = Base.dyn_cast<const ValueDecl*>();
    Info.Diag(Loc, diag::note_constexpr_past_end, 1)
      << !Designator.Entries.empty() << !!VD << VD;
    NoteLValueLocation(Info, Base);
  }

  return true;
}

/// Check that this core constant expression is of literal type, and if not,
/// produce an appropriate diagnostic.
static bool CheckLiteralType(EvalInfo &Info, const Expr *E) {
  if (!E->isRValue() || E->getType()->isLiteralType())
    return true;

  // Prvalue constant expressions must be of literal types.
  if (Info.getLangOpts().CPlusPlus0x)
    Info.Diag(E, diag::note_constexpr_nonliteral)
      << E->getType();
  else
    Info.Diag(E, diag::note_invalid_subexpr_in_const_expr);
  return false;
}

/// Check that this core constant expression value is a valid value for a
/// constant expression. If not, report an appropriate diagnostic. Does not
/// check that the expression is of literal type.
static bool CheckConstantExpression(EvalInfo &Info, SourceLocation DiagLoc,
                                    QualType Type, const APValue &Value) {
  // Core issue 1454: For a literal constant expression of array or class type,
  // each subobject of its value shall have been initialized by a constant
  // expression.
  if (Value.isArray()) {
    QualType EltTy = Type->castAsArrayTypeUnsafe()->getElementType();
    for (unsigned I = 0, N = Value.getArrayInitializedElts(); I != N; ++I) {
      if (!CheckConstantExpression(Info, DiagLoc, EltTy,
                                   Value.getArrayInitializedElt(I)))
        return false;
    }
    if (!Value.hasArrayFiller())
      return true;
    return CheckConstantExpression(Info, DiagLoc, EltTy,
                                   Value.getArrayFiller());
  }
  if (Value.isUnion() && Value.getUnionField()) {
    return CheckConstantExpression(Info, DiagLoc,
                                   Value.getUnionField()->getType(),
                                   Value.getUnionValue());
  }
  if (Value.isStruct()) {
    RecordDecl *RD = Type->castAs<RecordType>()->getDecl();
    if (const CXXRecordDecl *CD = dyn_cast<CXXRecordDecl>(RD)) {
      unsigned BaseIndex = 0;
      for (CXXRecordDecl::base_class_const_iterator I = CD->bases_begin(),
             End = CD->bases_end(); I != End; ++I, ++BaseIndex) {
        if (!CheckConstantExpression(Info, DiagLoc, I->getType(),
                                     Value.getStructBase(BaseIndex)))
          return false;
      }
    }
    for (RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
         I != E; ++I) {
      if (!CheckConstantExpression(Info, DiagLoc, I->getType(),
                                   Value.getStructField(I->getFieldIndex())))
        return false;
    }
  }

  if (Value.isLValue()) {
    LValue LVal;
    LVal.setFrom(Info.Ctx, Value);
    return CheckLValueConstantExpression(Info, DiagLoc, Type, LVal);
  }

  // Everything else is fine.
  return true;
}

const ValueDecl *GetLValueBaseDecl(const LValue &LVal) {
  return LVal.Base.dyn_cast<const ValueDecl*>();
}

static bool IsLiteralLValue(const LValue &Value) {
  return Value.Base.dyn_cast<const Expr*>() && !Value.CallIndex;
}

static bool IsWeakLValue(const LValue &Value) {
  const ValueDecl *Decl = GetLValueBaseDecl(Value);
  return Decl && Decl->isWeak();
}

static bool EvalPointerValueAsBool(const APValue &Value, bool &Result) {
  // A null base expression indicates a null pointer.  These are always
  // evaluatable, and they are false unless the offset is zero.
  if (!Value.getLValueBase()) {
    Result = !Value.getLValueOffset().isZero();
    return true;
  }

  // We have a non-null base.  These are generally known to be true, but if it's
  // a weak declaration it can be null at runtime.
  Result = true;
  const ValueDecl *Decl = Value.getLValueBase().dyn_cast<const ValueDecl*>();
  return !Decl || !Decl->isWeak();
}

static bool HandleConversionToBool(const APValue &Val, bool &Result) {
  switch (Val.getKind()) {
  case APValue::Uninitialized:
    return false;
  case APValue::Int:
    Result = Val.getInt().getBoolValue();
    return true;
  case APValue::Float:
    Result = !Val.getFloat().isZero();
    return true;
  case APValue::ComplexInt:
    Result = Val.getComplexIntReal().getBoolValue() ||
             Val.getComplexIntImag().getBoolValue();
    return true;
  case APValue::ComplexFloat:
    Result = !Val.getComplexFloatReal().isZero() ||
             !Val.getComplexFloatImag().isZero();
    return true;
  case APValue::LValue:
    return EvalPointerValueAsBool(Val, Result);
  case APValue::MemberPointer:
    Result = Val.getMemberPointerDecl();
    return true;
  case APValue::Vector:
  case APValue::Array:
  case APValue::Struct:
  case APValue::Union:
  case APValue::AddrLabelDiff:
    return false;
  }

  llvm_unreachable("unknown APValue kind");
}

static bool EvaluateAsBooleanCondition(const Expr *E, bool &Result,
                                       EvalInfo &Info) {
  assert(E->isRValue() && "missing lvalue-to-rvalue conv in bool condition");
  APValue Val;
  if (!Evaluate(Val, Info, E))
    return false;
  return HandleConversionToBool(Val, Result);
}

template<typename T>
static void HandleOverflow(EvalInfo &Info, const Expr *E,
                           const T &SrcValue, QualType DestType) {
  Info.CCEDiag(E, diag::note_constexpr_overflow)
    << SrcValue << DestType;
}

static bool HandleFloatToIntCast(EvalInfo &Info, const Expr *E,
                                 QualType SrcType, const APFloat &Value,
                                 QualType DestType, APSInt &Result) {
  unsigned DestWidth = Info.Ctx.getIntWidth(DestType);
  // Determine whether we are converting to unsigned or signed.
  bool DestSigned = DestType->isSignedIntegerOrEnumerationType();

  Result = APSInt(DestWidth, !DestSigned);
  bool ignored;
  if (Value.convertToInteger(Result, llvm::APFloat::rmTowardZero, &ignored)
      & APFloat::opInvalidOp)
    HandleOverflow(Info, E, Value, DestType);
  return true;
}

static bool HandleFloatToFloatCast(EvalInfo &Info, const Expr *E,
                                   QualType SrcType, QualType DestType,
                                   APFloat &Result) {
  APFloat Value = Result;
  bool ignored;
  if (Result.convert(Info.Ctx.getFloatTypeSemantics(DestType),
                     APFloat::rmNearestTiesToEven, &ignored)
      & APFloat::opOverflow)
    HandleOverflow(Info, E, Value, DestType);
  return true;
}

static APSInt HandleIntToIntCast(EvalInfo &Info, const Expr *E,
                                 QualType DestType, QualType SrcType,
                                 APSInt &Value) {
  unsigned DestWidth = Info.Ctx.getIntWidth(DestType);
  APSInt Result = Value;
  // Figure out if this is a truncate, extend or noop cast.
  // If the input is signed, do a sign extend, noop, or truncate.
  Result = Result.extOrTrunc(DestWidth);
  Result.setIsUnsigned(DestType->isUnsignedIntegerOrEnumerationType());
  return Result;
}

static bool HandleIntToFloatCast(EvalInfo &Info, const Expr *E,
                                 QualType SrcType, const APSInt &Value,
                                 QualType DestType, APFloat &Result) {
  Result = APFloat(Info.Ctx.getFloatTypeSemantics(DestType), 1);
  if (Result.convertFromAPInt(Value, Value.isSigned(),
                              APFloat::rmNearestTiesToEven)
      & APFloat::opOverflow)
    HandleOverflow(Info, E, Value, DestType);
  return true;
}

static bool EvalAndBitcastToAPInt(EvalInfo &Info, const Expr *E,
                                  llvm::APInt &Res) {
  APValue SVal;
  if (!Evaluate(SVal, Info, E))
    return false;
  if (SVal.isInt()) {
    Res = SVal.getInt();
    return true;
  }
  if (SVal.isFloat()) {
    Res = SVal.getFloat().bitcastToAPInt();
    return true;
  }
  if (SVal.isVector()) {
    QualType VecTy = E->getType();
    unsigned VecSize = Info.Ctx.getTypeSize(VecTy);
    QualType EltTy = VecTy->castAs<VectorType>()->getElementType();
    unsigned EltSize = Info.Ctx.getTypeSize(EltTy);
    bool BigEndian = Info.Ctx.getTargetInfo().isBigEndian();
    Res = llvm::APInt::getNullValue(VecSize);
    for (unsigned i = 0; i < SVal.getVectorLength(); i++) {
      APValue &Elt = SVal.getVectorElt(i);
      llvm::APInt EltAsInt;
      if (Elt.isInt()) {
        EltAsInt = Elt.getInt();
      } else if (Elt.isFloat()) {
        EltAsInt = Elt.getFloat().bitcastToAPInt();
      } else {
        // Don't try to handle vectors of anything other than int or float
        // (not sure if it's possible to hit this case).
        Info.Diag(E, diag::note_invalid_subexpr_in_const_expr);
        return false;
      }
      unsigned BaseEltSize = EltAsInt.getBitWidth();
      if (BigEndian)
        Res |= EltAsInt.zextOrTrunc(VecSize).rotr(i*EltSize+BaseEltSize);
      else
        Res |= EltAsInt.zextOrTrunc(VecSize).rotl(i*EltSize);
    }
    return true;
  }
  // Give up if the input isn't an int, float, or vector.  For example, we
  // reject "(v4i16)(intptr_t)&a".
  Info.Diag(E, diag::note_invalid_subexpr_in_const_expr);
  return false;
}

/// Cast an lvalue referring to a base subobject to a derived class, by
/// truncating the lvalue's path to the given length.
static bool CastToDerivedClass(EvalInfo &Info, const Expr *E, LValue &Result,
                               const RecordDecl *TruncatedType,
                               unsigned TruncatedElements) {
  SubobjectDesignator &D = Result.Designator;

  // Check we actually point to a derived class object.
  if (TruncatedElements == D.Entries.size())
    return true;
  assert(TruncatedElements >= D.MostDerivedPathLength &&
         "not casting to a derived class");
  if (!Result.checkSubobject(Info, E, CSK_Derived))
    return false;

  // Truncate the path to the subobject, and remove any derived-to-base offsets.
  const RecordDecl *RD = TruncatedType;
  for (unsigned I = TruncatedElements, N = D.Entries.size(); I != N; ++I) {
    if (RD->isInvalidDecl()) return false;
    const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(RD);
    const CXXRecordDecl *Base = getAsBaseClass(D.Entries[I]);
    if (isVirtualBaseClass(D.Entries[I]))
      Result.Offset -= Layout.getVBaseClassOffset(Base);
    else
      Result.Offset -= Layout.getBaseClassOffset(Base);
    RD = Base;
  }
  D.Entries.resize(TruncatedElements);
  return true;
}

static bool HandleLValueDirectBase(EvalInfo &Info, const Expr *E, LValue &Obj,
                                   const CXXRecordDecl *Derived,
                                   const CXXRecordDecl *Base,
                                   const ASTRecordLayout *RL = 0) {
  if (!RL) {
    if (Derived->isInvalidDecl()) return false;
    RL = &Info.Ctx.getASTRecordLayout(Derived);
  }

  Obj.getLValueOffset() += RL->getBaseClassOffset(Base);
  Obj.addDecl(Info, E, Base, /*Virtual*/ false);
  return true;
}

static bool HandleLValueBase(EvalInfo &Info, const Expr *E, LValue &Obj,
                             const CXXRecordDecl *DerivedDecl,
                             const CXXBaseSpecifier *Base) {
  const CXXRecordDecl *BaseDecl = Base->getType()->getAsCXXRecordDecl();

  if (!Base->isVirtual())
    return HandleLValueDirectBase(Info, E, Obj, DerivedDecl, BaseDecl);

  SubobjectDesignator &D = Obj.Designator;
  if (D.Invalid)
    return false;

  // Extract most-derived object and corresponding type.
  DerivedDecl = D.MostDerivedType->getAsCXXRecordDecl();
  if (!CastToDerivedClass(Info, E, Obj, DerivedDecl, D.MostDerivedPathLength))
    return false;

  // Find the virtual base class.
  if (DerivedDecl->isInvalidDecl()) return false;
  const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(DerivedDecl);
  Obj.getLValueOffset() += Layout.getVBaseClassOffset(BaseDecl);
  Obj.addDecl(Info, E, BaseDecl, /*Virtual*/ true);
  return true;
}

/// Update LVal to refer to the given field, which must be a member of the type
/// currently described by LVal.
static bool HandleLValueMember(EvalInfo &Info, const Expr *E, LValue &LVal,
                               const FieldDecl *FD,
                               const ASTRecordLayout *RL = 0) {
  if (!RL) {
    if (FD->getParent()->isInvalidDecl()) return false;
    RL = &Info.Ctx.getASTRecordLayout(FD->getParent());
  }

  unsigned I = FD->getFieldIndex();
  LVal.Offset += Info.Ctx.toCharUnitsFromBits(RL->getFieldOffset(I));
  LVal.addDecl(Info, E, FD);
  return true;
}

/// Update LVal to refer to the given indirect field.
static bool HandleLValueIndirectMember(EvalInfo &Info, const Expr *E,
                                       LValue &LVal,
                                       const IndirectFieldDecl *IFD) {
  for (IndirectFieldDecl::chain_iterator C = IFD->chain_begin(),
                                         CE = IFD->chain_end(); C != CE; ++C)
    if (!HandleLValueMember(Info, E, LVal, cast<FieldDecl>(*C)))
      return false;
  return true;
}

/// Get the size of the given type in char units.
static bool HandleSizeof(EvalInfo &Info, SourceLocation Loc,
                         QualType Type, CharUnits &Size) {
  // sizeof(void), __alignof__(void), sizeof(function) = 1 as a gcc
  // extension.
  if (Type->isVoidType() || Type->isFunctionType()) {
    Size = CharUnits::One();
    return true;
  }

  if (!Type->isConstantSizeType()) {
    // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
    // FIXME: Better diagnostic.
    Info.Diag(Loc);
    return false;
  }

  Size = Info.Ctx.getTypeSizeInChars(Type);
  return true;
}

/// Update a pointer value to model pointer arithmetic.
/// \param Info - Information about the ongoing evaluation.
/// \param E - The expression being evaluated, for diagnostic purposes.
/// \param LVal - The pointer value to be updated.
/// \param EltTy - The pointee type represented by LVal.
/// \param Adjustment - The adjustment, in objects of type EltTy, to add.
static bool HandleLValueArrayAdjustment(EvalInfo &Info, const Expr *E,
                                        LValue &LVal, QualType EltTy,
                                        int64_t Adjustment) {
  CharUnits SizeOfPointee;
  if (!HandleSizeof(Info, E->getExprLoc(), EltTy, SizeOfPointee))
    return false;

  // Compute the new offset in the appropriate width.
  LVal.Offset += Adjustment * SizeOfPointee;
  LVal.adjustIndex(Info, E, Adjustment);
  return true;
}

/// Update an lvalue to refer to a component of a complex number.
/// \param Info - Information about the ongoing evaluation.
/// \param LVal - The lvalue to be updated.
/// \param EltTy - The complex number's component type.
/// \param Imag - False for the real component, true for the imaginary.
static bool HandleLValueComplexElement(EvalInfo &Info, const Expr *E,
                                       LValue &LVal, QualType EltTy,
                                       bool Imag) {
  if (Imag) {
    CharUnits SizeOfComponent;
    if (!HandleSizeof(Info, E->getExprLoc(), EltTy, SizeOfComponent))
      return false;
    LVal.Offset += SizeOfComponent;
  }
  LVal.addComplex(Info, E, EltTy, Imag);
  return true;
}

/// Try to evaluate the initializer for a variable declaration.
static bool EvaluateVarDeclInit(EvalInfo &Info, const Expr *E,
                                const VarDecl *VD,
                                CallStackFrame *Frame, APValue &Result) {
  // If this is a parameter to an active constexpr function call, perform
  // argument substitution.
  if (const ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(VD)) {
    // Assume arguments of a potential constant expression are unknown
    // constant expressions.
    if (Info.CheckingPotentialConstantExpression)
      return false;
    if (!Frame || !Frame->Arguments) {
      Info.Diag(E, diag::note_invalid_subexpr_in_const_expr);
      return false;
    }
    Result = Frame->Arguments[PVD->getFunctionScopeIndex()];
    return true;
  }

  // Dig out the initializer, and use the declaration which it's attached to.
  const Expr *Init = VD->getAnyInitializer(VD);
  if (!Init || Init->isValueDependent()) {
    // If we're checking a potential constant expression, the variable could be
    // initialized later.
    if (!Info.CheckingPotentialConstantExpression)
      Info.Diag(E, diag::note_invalid_subexpr_in_const_expr);
    return false;
  }

  // If we're currently evaluating the initializer of this declaration, use that
  // in-flight value.
  if (Info.EvaluatingDecl == VD) {
    Result = *Info.EvaluatingDeclValue;
    return !Result.isUninit();
  }

  // Never evaluate the initializer of a weak variable. We can't be sure that
  // this is the definition which will be used.
  if (VD->isWeak()) {
    Info.Diag(E, diag::note_invalid_subexpr_in_const_expr);
    return false;
  }

  // Check that we can fold the initializer. In C++, we will have already done
  // this in the cases where it matters for conformance.
  llvm::SmallVector<PartialDiagnosticAt, 8> Notes;
  if (!VD->evaluateValue(Notes)) {
    Info.Diag(E, diag::note_constexpr_var_init_non_constant,
              Notes.size() + 1) << VD;
    Info.Note(VD->getLocation(), diag::note_declared_at);
    Info.addNotes(Notes);
    return false;
  } else if (!VD->checkInitIsICE()) {
    Info.CCEDiag(E, diag::note_constexpr_var_init_non_constant,
                 Notes.size() + 1) << VD;
    Info.Note(VD->getLocation(), diag::note_declared_at);
    Info.addNotes(Notes);
  }

  Result = *VD->getEvaluatedValue();
  return true;
}

static bool IsConstNonVolatile(QualType T) {
  Qualifiers Quals = T.getQualifiers();
  return Quals.hasConst() && !Quals.hasVolatile();
}

/// Get the base index of the given base class within an APValue representing
/// the given derived class.
static unsigned getBaseIndex(const CXXRecordDecl *Derived,
                             const CXXRecordDecl *Base) {
  Base = Base->getCanonicalDecl();
  unsigned Index = 0;
  for (CXXRecordDecl::base_class_const_iterator I = Derived->bases_begin(),
         E = Derived->bases_end(); I != E; ++I, ++Index) {
    if (I->getType()->getAsCXXRecordDecl()->getCanonicalDecl() == Base)
      return Index;
  }

  llvm_unreachable("base class missing from derived class's bases list");
}

/// Extract the value of a character from a string literal. CharType is used to
/// determine the expected signedness of the result -- a string literal used to
/// initialize an array of 'signed char' or 'unsigned char' might contain chars
/// of the wrong signedness.
static APSInt ExtractStringLiteralCharacter(EvalInfo &Info, const Expr *Lit,
                                            uint64_t Index, QualType CharType) {
  // FIXME: Support PredefinedExpr, ObjCEncodeExpr, MakeStringConstant
  const StringLiteral *S = dyn_cast<StringLiteral>(Lit);
  assert(S && "unexpected string literal expression kind");
  assert(CharType->isIntegerType() && "unexpected character type");

  APSInt Value(S->getCharByteWidth() * Info.Ctx.getCharWidth(),
               CharType->isUnsignedIntegerType());
  if (Index < S->getLength())
    Value = S->getCodeUnit(Index);
  return Value;
}

/// Extract the designated sub-object of an rvalue.
static bool ExtractSubobject(EvalInfo &Info, const Expr *E,
                             APValue &Obj, QualType ObjType,
                             const SubobjectDesignator &Sub, QualType SubType) {
  if (Sub.Invalid)
    // A diagnostic will have already been produced.
    return false;
  if (Sub.isOnePastTheEnd()) {
    Info.Diag(E, Info.getLangOpts().CPlusPlus0x ?
                (unsigned)diag::note_constexpr_read_past_end :
                (unsigned)diag::note_invalid_subexpr_in_const_expr);
    return false;
  }
  if (Sub.Entries.empty())
    return true;
  if (Info.CheckingPotentialConstantExpression && Obj.isUninit())
    // This object might be initialized later.
    return false;

  APValue *O = &Obj;
  // Walk the designator's path to find the subobject.
  for (unsigned I = 0, N = Sub.Entries.size(); I != N; ++I) {
    if (ObjType->isArrayType()) {
      // Next subobject is an array element.
      const ConstantArrayType *CAT = Info.Ctx.getAsConstantArrayType(ObjType);
      assert(CAT && "vla in literal type?");
      uint64_t Index = Sub.Entries[I].ArrayIndex;
      if (CAT->getSize().ule(Index)) {
        // Note, it should not be possible to form a pointer with a valid
        // designator which points more than one past the end of the array.
        Info.Diag(E, Info.getLangOpts().CPlusPlus0x ?
                    (unsigned)diag::note_constexpr_read_past_end :
                    (unsigned)diag::note_invalid_subexpr_in_const_expr);
        return false;
      }
      // An array object is represented as either an Array APValue or as an
      // LValue which refers to a string literal.
      if (O->isLValue()) {
        assert(I == N - 1 && "extracting subobject of character?");
        assert(!O->hasLValuePath() || O->getLValuePath().empty());
        Obj = APValue(ExtractStringLiteralCharacter(
          Info, O->getLValueBase().get<const Expr*>(), Index, SubType));
        return true;
      } else if (O->getArrayInitializedElts() > Index)
        O = &O->getArrayInitializedElt(Index);
      else
        O = &O->getArrayFiller();
      ObjType = CAT->getElementType();
    } else if (ObjType->isAnyComplexType()) {
      // Next subobject is a complex number.
      uint64_t Index = Sub.Entries[I].ArrayIndex;
      if (Index > 1) {
        Info.Diag(E, Info.getLangOpts().CPlusPlus0x ?
                    (unsigned)diag::note_constexpr_read_past_end :
                    (unsigned)diag::note_invalid_subexpr_in_const_expr);
        return false;
      }
      assert(I == N - 1 && "extracting subobject of scalar?");
      if (O->isComplexInt()) {
        Obj = APValue(Index ? O->getComplexIntImag()
                            : O->getComplexIntReal());
      } else {
        assert(O->isComplexFloat());
        Obj = APValue(Index ? O->getComplexFloatImag()
                            : O->getComplexFloatReal());
      }
      return true;
    } else if (const FieldDecl *Field = getAsField(Sub.Entries[I])) {
      if (Field->isMutable()) {
        Info.Diag(E, diag::note_constexpr_ltor_mutable, 1)
          << Field;
        Info.Note(Field->getLocation(), diag::note_declared_at);
        return false;
      }

      // Next subobject is a class, struct or union field.
      RecordDecl *RD = ObjType->castAs<RecordType>()->getDecl();
      if (RD->isUnion()) {
        const FieldDecl *UnionField = O->getUnionField();
        if (!UnionField ||
            UnionField->getCanonicalDecl() != Field->getCanonicalDecl()) {
          Info.Diag(E, diag::note_constexpr_read_inactive_union_member)
            << Field << !UnionField << UnionField;
          return false;
        }
        O = &O->getUnionValue();
      } else
        O = &O->getStructField(Field->getFieldIndex());
      ObjType = Field->getType();

      if (ObjType.isVolatileQualified()) {
        if (Info.getLangOpts().CPlusPlus) {
          // FIXME: Include a description of the path to the volatile subobject.
          Info.Diag(E, diag::note_constexpr_ltor_volatile_obj, 1)
            << 2 << Field;
          Info.Note(Field->getLocation(), diag::note_declared_at);
        } else {
          Info.Diag(E, diag::note_invalid_subexpr_in_const_expr);
        }
        return false;
      }
    } else {
      // Next subobject is a base class.
      const CXXRecordDecl *Derived = ObjType->getAsCXXRecordDecl();
      const CXXRecordDecl *Base = getAsBaseClass(Sub.Entries[I]);
      O = &O->getStructBase(getBaseIndex(Derived, Base));
      ObjType = Info.Ctx.getRecordType(Base);
    }

    if (O->isUninit()) {
      if (!Info.CheckingPotentialConstantExpression)
        Info.Diag(E, diag::note_constexpr_read_uninit);
      return false;
    }
  }

  // This may look super-stupid, but it serves an important purpose: if we just
  // swapped Obj and *O, we'd create an object which had itself as a subobject.
  // To avoid the leak, we ensure that Tmp ends up owning the original complete
  // object, which is destroyed by Tmp's destructor.
  APValue Tmp;
  O->swap(Tmp);
  Obj.swap(Tmp);
  return true;
}

/// Find the position where two subobject designators diverge, or equivalently
/// the length of the common initial subsequence.
static unsigned FindDesignatorMismatch(QualType ObjType,
                                       const SubobjectDesignator &A,
                                       const SubobjectDesignator &B,
                                       bool &WasArrayIndex) {
  unsigned I = 0, N = std::min(A.Entries.size(), B.Entries.size());
  for (/**/; I != N; ++I) {
    if (!ObjType.isNull() &&
        (ObjType->isArrayType() || ObjType->isAnyComplexType())) {
      // Next subobject is an array element.
      if (A.Entries[I].ArrayIndex != B.Entries[I].ArrayIndex) {
        WasArrayIndex = true;
        return I;
      }
      if (ObjType->isAnyComplexType())
        ObjType = ObjType->castAs<ComplexType>()->getElementType();
      else
        ObjType = ObjType->castAsArrayTypeUnsafe()->getElementType();
    } else {
      if (A.Entries[I].BaseOrMember != B.Entries[I].BaseOrMember) {
        WasArrayIndex = false;
        return I;
      }
      if (const FieldDecl *FD = getAsField(A.Entries[I]))
        // Next subobject is a field.
        ObjType = FD->getType();
      else
        // Next subobject is a base class.
        ObjType = QualType();
    }
  }
  WasArrayIndex = false;
  return I;
}

/// Determine whether the given subobject designators refer to elements of the
/// same array object.
static bool AreElementsOfSameArray(QualType ObjType,
                                   const SubobjectDesignator &A,
                                   const SubobjectDesignator &B) {
  if (A.Entries.size() != B.Entries.size())
    return false;

  bool IsArray = A.MostDerivedArraySize != 0;
  if (IsArray && A.MostDerivedPathLength != A.Entries.size())
    // A is a subobject of the array element.
    return false;

  // If A (and B) designates an array element, the last entry will be the array
  // index. That doesn't have to match. Otherwise, we're in the 'implicit array
  // of length 1' case, and the entire path must match.
  bool WasArrayIndex;
  unsigned CommonLength = FindDesignatorMismatch(ObjType, A, B, WasArrayIndex);
  return CommonLength >= A.Entries.size() - IsArray;
}

/// HandleLValueToRValueConversion - Perform an lvalue-to-rvalue conversion on
/// the given lvalue. This can also be used for 'lvalue-to-lvalue' conversions
/// for looking up the glvalue referred to by an entity of reference type.
///
/// \param Info - Information about the ongoing evaluation.
/// \param Conv - The expression for which we are performing the conversion.
///               Used for diagnostics.
/// \param Type - The type we expect this conversion to produce, before
///               stripping cv-qualifiers in the case of a non-clas type.
/// \param LVal - The glvalue on which we are attempting to perform this action.
/// \param RVal - The produced value will be placed here.
static bool HandleLValueToRValueConversion(EvalInfo &Info, const Expr *Conv,
                                           QualType Type,
                                           const LValue &LVal, APValue &RVal) {
  if (LVal.Designator.Invalid)
    // A diagnostic will have already been produced.
    return false;

  const Expr *Base = LVal.Base.dyn_cast<const Expr*>();

  if (!LVal.Base) {
    // FIXME: Indirection through a null pointer deserves a specific diagnostic.
    Info.Diag(Conv, diag::note_invalid_subexpr_in_const_expr);
    return false;
  }

  CallStackFrame *Frame = 0;
  if (LVal.CallIndex) {
    Frame = Info.getCallFrame(LVal.CallIndex);
    if (!Frame) {
      Info.Diag(Conv, diag::note_constexpr_lifetime_ended, 1) << !Base;
      NoteLValueLocation(Info, LVal.Base);
      return false;
    }
  }

  // C++11 DR1311: An lvalue-to-rvalue conversion on a volatile-qualified type
  // is not a constant expression (even if the object is non-volatile). We also
  // apply this rule to C++98, in order to conform to the expected 'volatile'
  // semantics.
  if (Type.isVolatileQualified()) {
    if (Info.getLangOpts().CPlusPlus)
      Info.Diag(Conv, diag::note_constexpr_ltor_volatile_type) << Type;
    else
      Info.Diag(Conv);
    return false;
  }

  if (const ValueDecl *D = LVal.Base.dyn_cast<const ValueDecl*>()) {
    // In C++98, const, non-volatile integers initialized with ICEs are ICEs.
    // In C++11, constexpr, non-volatile variables initialized with constant
    // expressions are constant expressions too. Inside constexpr functions,
    // parameters are constant expressions even if they're non-const.
    // In C, such things can also be folded, although they are not ICEs.
    const VarDecl *VD = dyn_cast<VarDecl>(D);
    if (VD) {
      if (const VarDecl *VDef = VD->getDefinition(Info.Ctx))
        VD = VDef;
    }
    if (!VD || VD->isInvalidDecl()) {
      Info.Diag(Conv);
      return false;
    }

    // DR1313: If the object is volatile-qualified but the glvalue was not,
    // behavior is undefined so the result is not a constant expression.
    QualType VT = VD->getType();
    if (VT.isVolatileQualified()) {
      if (Info.getLangOpts().CPlusPlus) {
        Info.Diag(Conv, diag::note_constexpr_ltor_volatile_obj, 1) << 1 << VD;
        Info.Note(VD->getLocation(), diag::note_declared_at);
      } else {
        Info.Diag(Conv);
      }
      return false;
    }

    if (!isa<ParmVarDecl>(VD)) {
      if (VD->isConstexpr()) {
        // OK, we can read this variable.
      } else if (VT->isIntegralOrEnumerationType()) {
        if (!VT.isConstQualified()) {
          if (Info.getLangOpts().CPlusPlus) {
            Info.Diag(Conv, diag::note_constexpr_ltor_non_const_int, 1) << VD;
            Info.Note(VD->getLocation(), diag::note_declared_at);
          } else {
            Info.Diag(Conv);
          }
          return false;
        }
      } else if (VT->isFloatingType() && VT.isConstQualified()) {
        // We support folding of const floating-point types, in order to make
        // static const data members of such types (supported as an extension)
        // more useful.
        if (Info.getLangOpts().CPlusPlus0x) {
          Info.CCEDiag(Conv, diag::note_constexpr_ltor_non_constexpr, 1) << VD;
          Info.Note(VD->getLocation(), diag::note_declared_at);
        } else {
          Info.CCEDiag(Conv);
        }
      } else {
        // FIXME: Allow folding of values of any literal type in all languages.
        if (Info.getLangOpts().CPlusPlus0x) {
          Info.Diag(Conv, diag::note_constexpr_ltor_non_constexpr, 1) << VD;
          Info.Note(VD->getLocation(), diag::note_declared_at);
        } else {
          Info.Diag(Conv);
        }
        return false;
      }
    }

    if (!EvaluateVarDeclInit(Info, Conv, VD, Frame, RVal))
      return false;

    if (isa<ParmVarDecl>(VD) || !VD->getAnyInitializer()->isLValue())
      return ExtractSubobject(Info, Conv, RVal, VT, LVal.Designator, Type);

    // The declaration was initialized by an lvalue, with no lvalue-to-rvalue
    // conversion. This happens when the declaration and the lvalue should be
    // considered synonymous, for instance when initializing an array of char
    // from a string literal. Continue as if the initializer lvalue was the
    // value we were originally given.
    assert(RVal.getLValueOffset().isZero() &&
           "offset for lvalue init of non-reference");
    Base = RVal.getLValueBase().get<const Expr*>();

    if (unsigned CallIndex = RVal.getLValueCallIndex()) {
      Frame = Info.getCallFrame(CallIndex);
      if (!Frame) {
        Info.Diag(Conv, diag::note_constexpr_lifetime_ended, 1) << !Base;
        NoteLValueLocation(Info, RVal.getLValueBase());
        return false;
      }
    } else {
      Frame = 0;
    }
  }

  // Volatile temporary objects cannot be read in constant expressions.
  if (Base->getType().isVolatileQualified()) {
    if (Info.getLangOpts().CPlusPlus) {
      Info.Diag(Conv, diag::note_constexpr_ltor_volatile_obj, 1) << 0;
      Info.Note(Base->getExprLoc(), diag::note_constexpr_temporary_here);
    } else {
      Info.Diag(Conv);
    }
    return false;
  }

  if (Frame) {
    // If this is a temporary expression with a nontrivial initializer, grab the
    // value from the relevant stack frame.
    RVal = Frame->Temporaries[Base];
  } else if (const CompoundLiteralExpr *CLE
             = dyn_cast<CompoundLiteralExpr>(Base)) {
    // In C99, a CompoundLiteralExpr is an lvalue, and we defer evaluating the
    // initializer until now for such expressions. Such an expression can't be
    // an ICE in C, so this only matters for fold.
    assert(!Info.getLangOpts().CPlusPlus && "lvalue compound literal in c++?");
    if (!Evaluate(RVal, Info, CLE->getInitializer()))
      return false;
  } else if (isa<StringLiteral>(Base)) {
    // We represent a string literal array as an lvalue pointing at the
    // corresponding expression, rather than building an array of chars.
    // FIXME: Support PredefinedExpr, ObjCEncodeExpr, MakeStringConstant
    RVal = APValue(Base, CharUnits::Zero(), APValue::NoLValuePath(), 0);
  } else {
    Info.Diag(Conv, diag::note_invalid_subexpr_in_const_expr);
    return false;
  }

  return ExtractSubobject(Info, Conv, RVal, Base->getType(), LVal.Designator,
                          Type);
}

/// Build an lvalue for the object argument of a member function call.
static bool EvaluateObjectArgument(EvalInfo &Info, const Expr *Object,
                                   LValue &This) {
  if (Object->getType()->isPointerType())
    return EvaluatePointer(Object, This, Info);

  if (Object->isGLValue())
    return EvaluateLValue(Object, This, Info);

  if (Object->getType()->isLiteralType())
    return EvaluateTemporary(Object, This, Info);

  return false;
}

/// HandleMemberPointerAccess - Evaluate a member access operation and build an
/// lvalue referring to the result.
///
/// \param Info - Information about the ongoing evaluation.
/// \param BO - The member pointer access operation.
/// \param LV - Filled in with a reference to the resulting object.
/// \param IncludeMember - Specifies whether the member itself is included in
///        the resulting LValue subobject designator. This is not possible when
///        creating a bound member function.
/// \return The field or method declaration to which the member pointer refers,
///         or 0 if evaluation fails.
static const ValueDecl *HandleMemberPointerAccess(EvalInfo &Info,
                                                  const BinaryOperator *BO,
                                                  LValue &LV,
                                                  bool IncludeMember = true) {
  assert(BO->getOpcode() == BO_PtrMemD || BO->getOpcode() == BO_PtrMemI);

  bool EvalObjOK = EvaluateObjectArgument(Info, BO->getLHS(), LV);
  if (!EvalObjOK && !Info.keepEvaluatingAfterFailure())
    return 0;

  MemberPtr MemPtr;
  if (!EvaluateMemberPointer(BO->getRHS(), MemPtr, Info))
    return 0;

  // C++11 [expr.mptr.oper]p6: If the second operand is the null pointer to
  // member value, the behavior is undefined.
  if (!MemPtr.getDecl())
    return 0;

  if (!EvalObjOK)
    return 0;

  if (MemPtr.isDerivedMember()) {
    // This is a member of some derived class. Truncate LV appropriately.
    // The end of the derived-to-base path for the base object must match the
    // derived-to-base path for the member pointer.
    if (LV.Designator.MostDerivedPathLength + MemPtr.Path.size() >
        LV.Designator.Entries.size())
      return 0;
    unsigned PathLengthToMember =
        LV.Designator.Entries.size() - MemPtr.Path.size();
    for (unsigned I = 0, N = MemPtr.Path.size(); I != N; ++I) {
      const CXXRecordDecl *LVDecl = getAsBaseClass(
          LV.Designator.Entries[PathLengthToMember + I]);
      const CXXRecordDecl *MPDecl = MemPtr.Path[I];
      if (LVDecl->getCanonicalDecl() != MPDecl->getCanonicalDecl())
        return 0;
    }

    // Truncate the lvalue to the appropriate derived class.
    if (!CastToDerivedClass(Info, BO, LV, MemPtr.getContainingRecord(),
                            PathLengthToMember))
      return 0;
  } else if (!MemPtr.Path.empty()) {
    // Extend the LValue path with the member pointer's path.
    LV.Designator.Entries.reserve(LV.Designator.Entries.size() +
                                  MemPtr.Path.size() + IncludeMember);

    // Walk down to the appropriate base class.
    QualType LVType = BO->getLHS()->getType();
    if (const PointerType *PT = LVType->getAs<PointerType>())
      LVType = PT->getPointeeType();
    const CXXRecordDecl *RD = LVType->getAsCXXRecordDecl();
    assert(RD && "member pointer access on non-class-type expression");
    // The first class in the path is that of the lvalue.
    for (unsigned I = 1, N = MemPtr.Path.size(); I != N; ++I) {
      const CXXRecordDecl *Base = MemPtr.Path[N - I - 1];
      if (!HandleLValueDirectBase(Info, BO, LV, RD, Base))
        return 0;
      RD = Base;
    }
    // Finally cast to the class containing the member.
    if (!HandleLValueDirectBase(Info, BO, LV, RD, MemPtr.getContainingRecord()))
      return 0;
  }

  // Add the member. Note that we cannot build bound member functions here.
  if (IncludeMember) {
    if (const FieldDecl *FD = dyn_cast<FieldDecl>(MemPtr.getDecl())) {
      if (!HandleLValueMember(Info, BO, LV, FD))
        return 0;
    } else if (const IndirectFieldDecl *IFD =
                 dyn_cast<IndirectFieldDecl>(MemPtr.getDecl())) {
      if (!HandleLValueIndirectMember(Info, BO, LV, IFD))
        return 0;
    } else {
      llvm_unreachable("can't construct reference to bound member function");
    }
  }

  return MemPtr.getDecl();
}

/// HandleBaseToDerivedCast - Apply the given base-to-derived cast operation on
/// the provided lvalue, which currently refers to the base object.
static bool HandleBaseToDerivedCast(EvalInfo &Info, const CastExpr *E,
                                    LValue &Result) {
  SubobjectDesignator &D = Result.Designator;
  if (D.Invalid || !Result.checkNullPointer(Info, E, CSK_Derived))
    return false;

  QualType TargetQT = E->getType();
  if (const PointerType *PT = TargetQT->getAs<PointerType>())
    TargetQT = PT->getPointeeType();

  // Check this cast lands within the final derived-to-base subobject path.
  if (D.MostDerivedPathLength + E->path_size() > D.Entries.size()) {
    Info.CCEDiag(E, diag::note_constexpr_invalid_downcast)
      << D.MostDerivedType << TargetQT;
    return false;
  }

  // Check the type of the final cast. We don't need to check the path,
  // since a cast can only be formed if the path is unique.
  unsigned NewEntriesSize = D.Entries.size() - E->path_size();
  const CXXRecordDecl *TargetType = TargetQT->getAsCXXRecordDecl();
  const CXXRecordDecl *FinalType;
  if (NewEntriesSize == D.MostDerivedPathLength)
    FinalType = D.MostDerivedType->getAsCXXRecordDecl();
  else
    FinalType = getAsBaseClass(D.Entries[NewEntriesSize - 1]);
  if (FinalType->getCanonicalDecl() != TargetType->getCanonicalDecl()) {
    Info.CCEDiag(E, diag::note_constexpr_invalid_downcast)
      << D.MostDerivedType << TargetQT;
    return false;
  }

  // Truncate the lvalue to the appropriate derived class.
  return CastToDerivedClass(Info, E, Result, TargetType, NewEntriesSize);
}

namespace {
enum EvalStmtResult {
  /// Evaluation failed.
  ESR_Failed,
  /// Hit a 'return' statement.
  ESR_Returned,
  /// Evaluation succeeded.
  ESR_Succeeded
};
}

// Evaluate a statement.
static EvalStmtResult EvaluateStmt(APValue &Result, EvalInfo &Info,
                                   const Stmt *S) {
  switch (S->getStmtClass()) {
  default:
    return ESR_Failed;

  case Stmt::NullStmtClass:
  case Stmt::DeclStmtClass:
    return ESR_Succeeded;

  case Stmt::ReturnStmtClass: {
    const Expr *RetExpr = cast<ReturnStmt>(S)->getRetValue();
    if (!Evaluate(Result, Info, RetExpr))
      return ESR_Failed;
    return ESR_Returned;
  }

  case Stmt::CompoundStmtClass: {
    const CompoundStmt *CS = cast<CompoundStmt>(S);
    for (CompoundStmt::const_body_iterator BI = CS->body_begin(),
           BE = CS->body_end(); BI != BE; ++BI) {
      EvalStmtResult ESR = EvaluateStmt(Result, Info, *BI);
      if (ESR != ESR_Succeeded)
        return ESR;
    }
    return ESR_Succeeded;
  }
  }
}

/// CheckTrivialDefaultConstructor - Check whether a constructor is a trivial
/// default constructor. If so, we'll fold it whether or not it's marked as
/// constexpr. If it is marked as constexpr, we will never implicitly define it,
/// so we need special handling.
static bool CheckTrivialDefaultConstructor(EvalInfo &Info, SourceLocation Loc,
                                           const CXXConstructorDecl *CD,
                                           bool IsValueInitialization) {
  if (!CD->isTrivial() || !CD->isDefaultConstructor())
    return false;

  // Value-initialization does not call a trivial default constructor, so such a
  // call is a core constant expression whether or not the constructor is
  // constexpr.
  if (!CD->isConstexpr() && !IsValueInitialization) {
    if (Info.getLangOpts().CPlusPlus0x) {
      // FIXME: If DiagDecl is an implicitly-declared special member function,
      // we should be much more explicit about why it's not constexpr.
      Info.CCEDiag(Loc, diag::note_constexpr_invalid_function, 1)
        << /*IsConstexpr*/0 << /*IsConstructor*/1 << CD;
      Info.Note(CD->getLocation(), diag::note_declared_at);
    } else {
      Info.CCEDiag(Loc, diag::note_invalid_subexpr_in_const_expr);
    }
  }
  return true;
}

/// CheckConstexprFunction - Check that a function can be called in a constant
/// expression.
static bool CheckConstexprFunction(EvalInfo &Info, SourceLocation CallLoc,
                                   const FunctionDecl *Declaration,
                                   const FunctionDecl *Definition) {
  // Potential constant expressions can contain calls to declared, but not yet
  // defined, constexpr functions.
  if (Info.CheckingPotentialConstantExpression && !Definition &&
      Declaration->isConstexpr())
    return false;

  // Can we evaluate this function call?
  if (Definition && Definition->isConstexpr() && !Definition->isInvalidDecl())
    return true;

  if (Info.getLangOpts().CPlusPlus0x) {
    const FunctionDecl *DiagDecl = Definition ? Definition : Declaration;
    // FIXME: If DiagDecl is an implicitly-declared special member function, we
    // should be much more explicit about why it's not constexpr.
    Info.Diag(CallLoc, diag::note_constexpr_invalid_function, 1)
      << DiagDecl->isConstexpr() << isa<CXXConstructorDecl>(DiagDecl)
      << DiagDecl;
    Info.Note(DiagDecl->getLocation(), diag::note_declared_at);
  } else {
    Info.Diag(CallLoc, diag::note_invalid_subexpr_in_const_expr);
  }
  return false;
}

namespace {
typedef SmallVector<APValue, 8> ArgVector;
}

/// EvaluateArgs - Evaluate the arguments to a function call.
static bool EvaluateArgs(ArrayRef<const Expr*> Args, ArgVector &ArgValues,
                         EvalInfo &Info) {
  bool Success = true;
  for (ArrayRef<const Expr*>::iterator I = Args.begin(), E = Args.end();
       I != E; ++I) {
    if (!Evaluate(ArgValues[I - Args.begin()], Info, *I)) {
      // If we're checking for a potential constant expression, evaluate all
      // initializers even if some of them fail.
      if (!Info.keepEvaluatingAfterFailure())
        return false;
      Success = false;
    }
  }
  return Success;
}

/// Evaluate a function call.
static bool HandleFunctionCall(SourceLocation CallLoc,
                               const FunctionDecl *Callee, const LValue *This,
                               ArrayRef<const Expr*> Args, const Stmt *Body,
                               EvalInfo &Info, APValue &Result) {
  ArgVector ArgValues(Args.size());
  if (!EvaluateArgs(Args, ArgValues, Info))
    return false;

  if (!Info.CheckCallLimit(CallLoc))
    return false;

  CallStackFrame Frame(Info, CallLoc, Callee, This, ArgValues.data());
  return EvaluateStmt(Result, Info, Body) == ESR_Returned;
}

/// Evaluate a constructor call.
static bool HandleConstructorCall(SourceLocation CallLoc, const LValue &This,
                                  ArrayRef<const Expr*> Args,
                                  const CXXConstructorDecl *Definition,
                                  EvalInfo &Info, APValue &Result) {
  ArgVector ArgValues(Args.size());
  if (!EvaluateArgs(Args, ArgValues, Info))
    return false;

  if (!Info.CheckCallLimit(CallLoc))
    return false;

  const CXXRecordDecl *RD = Definition->getParent();
  if (RD->getNumVBases()) {
    Info.Diag(CallLoc, diag::note_constexpr_virtual_base) << RD;
    return false;
  }

  CallStackFrame Frame(Info, CallLoc, Definition, &This, ArgValues.data());

  // If it's a delegating constructor, just delegate.
  if (Definition->isDelegatingConstructor()) {
    CXXConstructorDecl::init_const_iterator I = Definition->init_begin();
    return EvaluateInPlace(Result, Info, This, (*I)->getInit());
  }

  // For a trivial copy or move constructor, perform an APValue copy. This is
  // essential for unions, where the operations performed by the constructor
  // cannot be represented by ctor-initializers.
  if (Definition->isDefaulted() &&
      ((Definition->isCopyConstructor() && Definition->isTrivial()) ||
       (Definition->isMoveConstructor() && Definition->isTrivial()))) {
    LValue RHS;
    RHS.setFrom(Info.Ctx, ArgValues[0]);
    return HandleLValueToRValueConversion(Info, Args[0], Args[0]->getType(),
                                          RHS, Result);
  }

  // Reserve space for the struct members.
  if (!RD->isUnion() && Result.isUninit())
    Result = APValue(APValue::UninitStruct(), RD->getNumBases(),
                     std::distance(RD->field_begin(), RD->field_end()));

  if (RD->isInvalidDecl()) return false;
  const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(RD);

  bool Success = true;
  unsigned BasesSeen = 0;
#ifndef NDEBUG
  CXXRecordDecl::base_class_const_iterator BaseIt = RD->bases_begin();
#endif
  for (CXXConstructorDecl::init_const_iterator I = Definition->init_begin(),
       E = Definition->init_end(); I != E; ++I) {
    LValue Subobject = This;
    APValue *Value = &Result;

    // Determine the subobject to initialize.
    if ((*I)->isBaseInitializer()) {
      QualType BaseType((*I)->getBaseClass(), 0);
#ifndef NDEBUG
      // Non-virtual base classes are initialized in the order in the class
      // definition. We have already checked for virtual base classes.
      assert(!BaseIt->isVirtual() && "virtual base for literal type");
      assert(Info.Ctx.hasSameType(BaseIt->getType(), BaseType) &&
             "base class initializers not in expected order");
      ++BaseIt;
#endif
      if (!HandleLValueDirectBase(Info, (*I)->getInit(), Subobject, RD,
                                  BaseType->getAsCXXRecordDecl(), &Layout))
        return false;
      Value = &Result.getStructBase(BasesSeen++);
    } else if (FieldDecl *FD = (*I)->getMember()) {
      if (!HandleLValueMember(Info, (*I)->getInit(), Subobject, FD, &Layout))
        return false;
      if (RD->isUnion()) {
        Result = APValue(FD);
        Value = &Result.getUnionValue();
      } else {
        Value = &Result.getStructField(FD->getFieldIndex());
      }
    } else if (IndirectFieldDecl *IFD = (*I)->getIndirectMember()) {
      // Walk the indirect field decl's chain to find the object to initialize,
      // and make sure we've initialized every step along it.
      for (IndirectFieldDecl::chain_iterator C = IFD->chain_begin(),
                                             CE = IFD->chain_end();
           C != CE; ++C) {
        FieldDecl *FD = cast<FieldDecl>(*C);
        CXXRecordDecl *CD = cast<CXXRecordDecl>(FD->getParent());
        // Switch the union field if it differs. This happens if we had
        // preceding zero-initialization, and we're now initializing a union
        // subobject other than the first.
        // FIXME: In this case, the values of the other subobjects are
        // specified, since zero-initialization sets all padding bits to zero.
        if (Value->isUninit() ||
            (Value->isUnion() && Value->getUnionField() != FD)) {
          if (CD->isUnion())
            *Value = APValue(FD);
          else
            *Value = APValue(APValue::UninitStruct(), CD->getNumBases(),
                             std::distance(CD->field_begin(), CD->field_end()));
        }
        if (!HandleLValueMember(Info, (*I)->getInit(), Subobject, FD))
          return false;
        if (CD->isUnion())
          Value = &Value->getUnionValue();
        else
          Value = &Value->getStructField(FD->getFieldIndex());
      }
    } else {
      llvm_unreachable("unknown base initializer kind");
    }

    if (!EvaluateInPlace(*Value, Info, Subobject, (*I)->getInit(),
                         (*I)->isBaseInitializer()
                                      ? CCEK_Constant : CCEK_MemberInit)) {
      // If we're checking for a potential constant expression, evaluate all
      // initializers even if some of them fail.
      if (!Info.keepEvaluatingAfterFailure())
        return false;
      Success = false;
    }
  }

  return Success;
}

//===----------------------------------------------------------------------===//
// Generic Evaluation
//===----------------------------------------------------------------------===//
namespace {

// FIXME: RetTy is always bool. Remove it.
template <class Derived, typename RetTy=bool>
class ExprEvaluatorBase
  : public ConstStmtVisitor<Derived, RetTy> {
private:
  RetTy DerivedSuccess(const APValue &V, const Expr *E) {
    return static_cast<Derived*>(this)->Success(V, E);
  }
  RetTy DerivedZeroInitialization(const Expr *E) {
    return static_cast<Derived*>(this)->ZeroInitialization(E);
  }

  // Check whether a conditional operator with a non-constant condition is a
  // potential constant expression. If neither arm is a potential constant
  // expression, then the conditional operator is not either.
  template<typename ConditionalOperator>
  void CheckPotentialConstantConditional(const ConditionalOperator *E) {
    assert(Info.CheckingPotentialConstantExpression);

    // Speculatively evaluate both arms.
    {
      llvm::SmallVector<PartialDiagnosticAt, 8> Diag;
      SpeculativeEvaluationRAII Speculate(Info, &Diag);

      StmtVisitorTy::Visit(E->getFalseExpr());
      if (Diag.empty())
        return;

      Diag.clear();
      StmtVisitorTy::Visit(E->getTrueExpr());
      if (Diag.empty())
        return;
    }

    Error(E, diag::note_constexpr_conditional_never_const);
  }


  template<typename ConditionalOperator>
  bool HandleConditionalOperator(const ConditionalOperator *E) {
    bool BoolResult;
    if (!EvaluateAsBooleanCondition(E->getCond(), BoolResult, Info)) {
      if (Info.CheckingPotentialConstantExpression)
        CheckPotentialConstantConditional(E);
      return false;
    }

    Expr *EvalExpr = BoolResult ? E->getTrueExpr() : E->getFalseExpr();
    return StmtVisitorTy::Visit(EvalExpr);
  }

protected:
  EvalInfo &Info;
  typedef ConstStmtVisitor<Derived, RetTy> StmtVisitorTy;
  typedef ExprEvaluatorBase ExprEvaluatorBaseTy;

  OptionalDiagnostic CCEDiag(const Expr *E, diag::kind D) {
    return Info.CCEDiag(E, D);
  }

  RetTy ZeroInitialization(const Expr *E) { return Error(E); }

public:
  ExprEvaluatorBase(EvalInfo &Info) : Info(Info) {}

  EvalInfo &getEvalInfo() { return Info; }

  /// Report an evaluation error. This should only be called when an error is
  /// first discovered. When propagating an error, just return false.
  bool Error(const Expr *E, diag::kind D) {
    Info.Diag(E, D);
    return false;
  }
  bool Error(const Expr *E) {
    return Error(E, diag::note_invalid_subexpr_in_const_expr);
  }

  RetTy VisitStmt(const Stmt *) {
    llvm_unreachable("Expression evaluator should not be called on stmts");
  }
  RetTy VisitExpr(const Expr *E) {
    return Error(E);
  }

  RetTy VisitParenExpr(const ParenExpr *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }
  RetTy VisitUnaryExtension(const UnaryOperator *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }
  RetTy VisitUnaryPlus(const UnaryOperator *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }
  RetTy VisitChooseExpr(const ChooseExpr *E)
    { return StmtVisitorTy::Visit(E->getChosenSubExpr(Info.Ctx)); }
  RetTy VisitGenericSelectionExpr(const GenericSelectionExpr *E)
    { return StmtVisitorTy::Visit(E->getResultExpr()); }
  RetTy VisitSubstNonTypeTemplateParmExpr(const SubstNonTypeTemplateParmExpr *E)
    { return StmtVisitorTy::Visit(E->getReplacement()); }
  RetTy VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *E)
    { return StmtVisitorTy::Visit(E->getExpr()); }
  // We cannot create any objects for which cleanups are required, so there is
  // nothing to do here; all cleanups must come from unevaluated subexpressions.
  RetTy VisitExprWithCleanups(const ExprWithCleanups *E)
    { return StmtVisitorTy::Visit(E->getSubExpr()); }

  RetTy VisitCXXReinterpretCastExpr(const CXXReinterpretCastExpr *E) {
    CCEDiag(E, diag::note_constexpr_invalid_cast) << 0;
    return static_cast<Derived*>(this)->VisitCastExpr(E);
  }
  RetTy VisitCXXDynamicCastExpr(const CXXDynamicCastExpr *E) {
    CCEDiag(E, diag::note_constexpr_invalid_cast) << 1;
    return static_cast<Derived*>(this)->VisitCastExpr(E);
  }

  RetTy VisitBinaryOperator(const BinaryOperator *E) {
    switch (E->getOpcode()) {
    default:
      return Error(E);

    case BO_Comma:
      VisitIgnoredValue(E->getLHS());
      return StmtVisitorTy::Visit(E->getRHS());

    case BO_PtrMemD:
    case BO_PtrMemI: {
      LValue Obj;
      if (!HandleMemberPointerAccess(Info, E, Obj))
        return false;
      APValue Result;
      if (!HandleLValueToRValueConversion(Info, E, E->getType(), Obj, Result))
        return false;
      return DerivedSuccess(Result, E);
    }
    }
  }

  RetTy VisitBinaryConditionalOperator(const BinaryConditionalOperator *E) {
    // Evaluate and cache the common expression. We treat it as a temporary,
    // even though it's not quite the same thing.
    if (!Evaluate(Info.CurrentCall->Temporaries[E->getOpaqueValue()],
                  Info, E->getCommon()))
      return false;

    return HandleConditionalOperator(E);
  }

  RetTy VisitConditionalOperator(const ConditionalOperator *E) {
    bool IsBcpCall = false;
    // If the condition (ignoring parens) is a __builtin_constant_p call,
    // the result is a constant expression if it can be folded without
    // side-effects. This is an important GNU extension. See GCC PR38377
    // for discussion.
    if (const CallExpr *CallCE =
          dyn_cast<CallExpr>(E->getCond()->IgnoreParenCasts()))
      if (CallCE->isBuiltinCall() == Builtin::BI__builtin_constant_p)
        IsBcpCall = true;

    // Always assume __builtin_constant_p(...) ? ... : ... is a potential
    // constant expression; we can't check whether it's potentially foldable.
    if (Info.CheckingPotentialConstantExpression && IsBcpCall)
      return false;

    FoldConstant Fold(Info);

    if (!HandleConditionalOperator(E))
      return false;

    if (IsBcpCall)
      Fold.Fold(Info);

    return true;
  }

  RetTy VisitOpaqueValueExpr(const OpaqueValueExpr *E) {
    APValue &Value = Info.CurrentCall->Temporaries[E];
    if (Value.isUninit()) {
      const Expr *Source = E->getSourceExpr();
      if (!Source)
        return Error(E);
      if (Source == E) { // sanity checking.
        assert(0 && "OpaqueValueExpr recursively refers to itself");
        return Error(E);
      }
      return StmtVisitorTy::Visit(Source);
    }
    return DerivedSuccess(Value, E);
  }

  RetTy VisitCallExpr(const CallExpr *E) {
    const Expr *Callee = E->getCallee()->IgnoreParens();
    QualType CalleeType = Callee->getType();

    const FunctionDecl *FD = 0;
    LValue *This = 0, ThisVal;
    llvm::ArrayRef<const Expr*> Args(E->getArgs(), E->getNumArgs());
    bool HasQualifier = false;

    // Extract function decl and 'this' pointer from the callee.
    if (CalleeType->isSpecificBuiltinType(BuiltinType::BoundMember)) {
      const ValueDecl *Member = 0;
      if (const MemberExpr *ME = dyn_cast<MemberExpr>(Callee)) {
        // Explicit bound member calls, such as x.f() or p->g();
        if (!EvaluateObjectArgument(Info, ME->getBase(), ThisVal))
          return false;
        Member = ME->getMemberDecl();
        This = &ThisVal;
        HasQualifier = ME->hasQualifier();
      } else if (const BinaryOperator *BE = dyn_cast<BinaryOperator>(Callee)) {
        // Indirect bound member calls ('.*' or '->*').
        Member = HandleMemberPointerAccess(Info, BE, ThisVal, false);
        if (!Member) return false;
        This = &ThisVal;
      } else
        return Error(Callee);

      FD = dyn_cast<FunctionDecl>(Member);
      if (!FD)
        return Error(Callee);
    } else if (CalleeType->isFunctionPointerType()) {
      LValue Call;
      if (!EvaluatePointer(Callee, Call, Info))
        return false;

      if (!Call.getLValueOffset().isZero())
        return Error(Callee);
      FD = dyn_cast_or_null<FunctionDecl>(
                             Call.getLValueBase().dyn_cast<const ValueDecl*>());
      if (!FD)
        return Error(Callee);

      // Overloaded operator calls to member functions are represented as normal
      // calls with '*this' as the first argument.
      const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
      if (MD && !MD->isStatic()) {
        // FIXME: When selecting an implicit conversion for an overloaded
        // operator delete, we sometimes try to evaluate calls to conversion
        // operators without a 'this' parameter!
        if (Args.empty())
          return Error(E);

        if (!EvaluateObjectArgument(Info, Args[0], ThisVal))
          return false;
        This = &ThisVal;
        Args = Args.slice(1);
      }

      // Don't call function pointers which have been cast to some other type.
      if (!Info.Ctx.hasSameType(CalleeType->getPointeeType(), FD->getType()))
        return Error(E);
    } else
      return Error(E);

    if (This && !This->checkSubobject(Info, E, CSK_This))
      return false;

    // DR1358 allows virtual constexpr functions in some cases. Don't allow
    // calls to such functions in constant expressions.
    if (This && !HasQualifier &&
        isa<CXXMethodDecl>(FD) && cast<CXXMethodDecl>(FD)->isVirtual())
      return Error(E, diag::note_constexpr_virtual_call);

    const FunctionDecl *Definition = 0;
    Stmt *Body = FD->getBody(Definition);
    APValue Result;

    if (!CheckConstexprFunction(Info, E->getExprLoc(), FD, Definition) ||
        !HandleFunctionCall(E->getExprLoc(), Definition, This, Args, Body,
                            Info, Result))
      return false;

    return DerivedSuccess(Result, E);
  }

  RetTy VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
    return StmtVisitorTy::Visit(E->getInitializer());
  }
  RetTy VisitInitListExpr(const InitListExpr *E) {
    if (E->getNumInits() == 0)
      return DerivedZeroInitialization(E);
    if (E->getNumInits() == 1)
      return StmtVisitorTy::Visit(E->getInit(0));
    return Error(E);
  }
  RetTy VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    return DerivedZeroInitialization(E);
  }
  RetTy VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
    return DerivedZeroInitialization(E);
  }
  RetTy VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *E) {
    return DerivedZeroInitialization(E);
  }

  /// A member expression where the object is a prvalue is itself a prvalue.
  RetTy VisitMemberExpr(const MemberExpr *E) {
    assert(!E->isArrow() && "missing call to bound member function?");

    APValue Val;
    if (!Evaluate(Val, Info, E->getBase()))
      return false;

    QualType BaseTy = E->getBase()->getType();

    const FieldDecl *FD = dyn_cast<FieldDecl>(E->getMemberDecl());
    if (!FD) return Error(E);
    assert(!FD->getType()->isReferenceType() && "prvalue reference?");
    assert(BaseTy->castAs<RecordType>()->getDecl()->getCanonicalDecl() ==
           FD->getParent()->getCanonicalDecl() && "record / field mismatch");

    SubobjectDesignator Designator(BaseTy);
    Designator.addDeclUnchecked(FD);

    return ExtractSubobject(Info, E, Val, BaseTy, Designator, E->getType()) &&
           DerivedSuccess(Val, E);
  }

  RetTy VisitCastExpr(const CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      break;

    case CK_AtomicToNonAtomic:
    case CK_NonAtomicToAtomic:
    case CK_NoOp:
    case CK_UserDefinedConversion:
      return StmtVisitorTy::Visit(E->getSubExpr());

    case CK_LValueToRValue: {
      LValue LVal;
      if (!EvaluateLValue(E->getSubExpr(), LVal, Info))
        return false;
      APValue RVal;
      // Note, we use the subexpression's type in order to retain cv-qualifiers.
      if (!HandleLValueToRValueConversion(Info, E, E->getSubExpr()->getType(),
                                          LVal, RVal))
        return false;
      return DerivedSuccess(RVal, E);
    }
    }

    return Error(E);
  }

  /// Visit a value which is evaluated, but whose value is ignored.
  void VisitIgnoredValue(const Expr *E) {
    APValue Scratch;
    if (!Evaluate(Scratch, Info, E))
      Info.EvalStatus.HasSideEffects = true;
  }
};

}

//===----------------------------------------------------------------------===//
// Common base class for lvalue and temporary evaluation.
//===----------------------------------------------------------------------===//
namespace {
template<class Derived>
class LValueExprEvaluatorBase
  : public ExprEvaluatorBase<Derived, bool> {
protected:
  LValue &Result;
  typedef LValueExprEvaluatorBase LValueExprEvaluatorBaseTy;
  typedef ExprEvaluatorBase<Derived, bool> ExprEvaluatorBaseTy;

  bool Success(APValue::LValueBase B) {
    Result.set(B);
    return true;
  }

public:
  LValueExprEvaluatorBase(EvalInfo &Info, LValue &Result) :
    ExprEvaluatorBaseTy(Info), Result(Result) {}

  bool Success(const APValue &V, const Expr *E) {
    Result.setFrom(this->Info.Ctx, V);
    return true;
  }

  bool VisitMemberExpr(const MemberExpr *E) {
    // Handle non-static data members.
    QualType BaseTy;
    if (E->isArrow()) {
      if (!EvaluatePointer(E->getBase(), Result, this->Info))
        return false;
      BaseTy = E->getBase()->getType()->castAs<PointerType>()->getPointeeType();
    } else if (E->getBase()->isRValue()) {
      assert(E->getBase()->getType()->isRecordType());
      if (!EvaluateTemporary(E->getBase(), Result, this->Info))
        return false;
      BaseTy = E->getBase()->getType();
    } else {
      if (!this->Visit(E->getBase()))
        return false;
      BaseTy = E->getBase()->getType();
    }

    const ValueDecl *MD = E->getMemberDecl();
    if (const FieldDecl *FD = dyn_cast<FieldDecl>(E->getMemberDecl())) {
      assert(BaseTy->getAs<RecordType>()->getDecl()->getCanonicalDecl() ==
             FD->getParent()->getCanonicalDecl() && "record / field mismatch");
      (void)BaseTy;
      if (!HandleLValueMember(this->Info, E, Result, FD))
        return false;
    } else if (const IndirectFieldDecl *IFD = dyn_cast<IndirectFieldDecl>(MD)) {
      if (!HandleLValueIndirectMember(this->Info, E, Result, IFD))
        return false;
    } else
      return this->Error(E);

    if (MD->getType()->isReferenceType()) {
      APValue RefValue;
      if (!HandleLValueToRValueConversion(this->Info, E, MD->getType(), Result,
                                          RefValue))
        return false;
      return Success(RefValue, E);
    }
    return true;
  }

  bool VisitBinaryOperator(const BinaryOperator *E) {
    switch (E->getOpcode()) {
    default:
      return ExprEvaluatorBaseTy::VisitBinaryOperator(E);

    case BO_PtrMemD:
    case BO_PtrMemI:
      return HandleMemberPointerAccess(this->Info, E, Result);
    }
  }

  bool VisitCastExpr(const CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      return ExprEvaluatorBaseTy::VisitCastExpr(E);

    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase: {
      if (!this->Visit(E->getSubExpr()))
        return false;

      // Now figure out the necessary offset to add to the base LV to get from
      // the derived class to the base class.
      QualType Type = E->getSubExpr()->getType();

      for (CastExpr::path_const_iterator PathI = E->path_begin(),
           PathE = E->path_end(); PathI != PathE; ++PathI) {
        if (!HandleLValueBase(this->Info, E, Result, Type->getAsCXXRecordDecl(),
                              *PathI))
          return false;
        Type = (*PathI)->getType();
      }

      return true;
    }
    }
  }
};
}

//===----------------------------------------------------------------------===//
// LValue Evaluation
//
// This is used for evaluating lvalues (in C and C++), xvalues (in C++11),
// function designators (in C), decl references to void objects (in C), and
// temporaries (if building with -Wno-address-of-temporary).
//
// LValue evaluation produces values comprising a base expression of one of the
// following types:
// - Declarations
//  * VarDecl
//  * FunctionDecl
// - Literals
//  * CompoundLiteralExpr in C
//  * StringLiteral
//  * CXXTypeidExpr
//  * PredefinedExpr
//  * ObjCStringLiteralExpr
//  * ObjCEncodeExpr
//  * AddrLabelExpr
//  * BlockExpr
//  * CallExpr for a MakeStringConstant builtin
// - Locals and temporaries
//  * Any Expr, with a CallIndex indicating the function in which the temporary
//    was evaluated.
// plus an offset in bytes.
//===----------------------------------------------------------------------===//
namespace {
class LValueExprEvaluator
  : public LValueExprEvaluatorBase<LValueExprEvaluator> {
public:
  LValueExprEvaluator(EvalInfo &Info, LValue &Result) :
    LValueExprEvaluatorBaseTy(Info, Result) {}

  bool VisitVarDecl(const Expr *E, const VarDecl *VD);

  bool VisitDeclRefExpr(const DeclRefExpr *E);
  bool VisitPredefinedExpr(const PredefinedExpr *E) { return Success(E); }
  bool VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E);
  bool VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
  bool VisitMemberExpr(const MemberExpr *E);
  bool VisitStringLiteral(const StringLiteral *E) { return Success(E); }
  bool VisitObjCEncodeExpr(const ObjCEncodeExpr *E) { return Success(E); }
  bool VisitCXXTypeidExpr(const CXXTypeidExpr *E);
  bool VisitCXXUuidofExpr(const CXXUuidofExpr *E);
  bool VisitArraySubscriptExpr(const ArraySubscriptExpr *E);
  bool VisitUnaryDeref(const UnaryOperator *E);
  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

  bool VisitCastExpr(const CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      return LValueExprEvaluatorBaseTy::VisitCastExpr(E);

    case CK_LValueBitCast:
      this->CCEDiag(E, diag::note_constexpr_invalid_cast) << 2;
      if (!Visit(E->getSubExpr()))
        return false;
      Result.Designator.setInvalid();
      return true;

    case CK_BaseToDerived:
      if (!Visit(E->getSubExpr()))
        return false;
      return HandleBaseToDerivedCast(Info, E, Result);
    }
  }
};
} // end anonymous namespace

/// Evaluate an expression as an lvalue. This can be legitimately called on
/// expressions which are not glvalues, in a few cases:
///  * function designators in C,
///  * "extern void" objects,
///  * temporaries, if building with -Wno-address-of-temporary.
static bool EvaluateLValue(const Expr* E, LValue& Result, EvalInfo &Info) {
  assert((E->isGLValue() || E->getType()->isFunctionType() ||
          E->getType()->isVoidType() || isa<CXXTemporaryObjectExpr>(E)) &&
         "can't evaluate expression as an lvalue");
  return LValueExprEvaluator(Info, Result).Visit(E);
}

bool LValueExprEvaluator::VisitDeclRefExpr(const DeclRefExpr *E) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(E->getDecl()))
    return Success(FD);
  if (const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl()))
    return VisitVarDecl(E, VD);
  return Error(E);
}

bool LValueExprEvaluator::VisitVarDecl(const Expr *E, const VarDecl *VD) {
  if (!VD->getType()->isReferenceType()) {
    if (isa<ParmVarDecl>(VD)) {
      Result.set(VD, Info.CurrentCall->Index);
      return true;
    }
    return Success(VD);
  }

  APValue V;
  if (!EvaluateVarDeclInit(Info, E, VD, Info.CurrentCall, V))
    return false;
  return Success(V, E);
}

bool LValueExprEvaluator::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *E) {
  if (E->GetTemporaryExpr()->isRValue()) {
    if (E->getType()->isRecordType())
      return EvaluateTemporary(E->GetTemporaryExpr(), Result, Info);

    Result.set(E, Info.CurrentCall->Index);
    return EvaluateInPlace(Info.CurrentCall->Temporaries[E], Info,
                           Result, E->GetTemporaryExpr());
  }

  // Materialization of an lvalue temporary occurs when we need to force a copy
  // (for instance, if it's a bitfield).
  // FIXME: The AST should contain an lvalue-to-rvalue node for such cases.
  if (!Visit(E->GetTemporaryExpr()))
    return false;
  if (!HandleLValueToRValueConversion(Info, E, E->getType(), Result,
                                      Info.CurrentCall->Temporaries[E]))
    return false;
  Result.set(E, Info.CurrentCall->Index);
  return true;
}

bool
LValueExprEvaluator::VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
  assert(!Info.getLangOpts().CPlusPlus && "lvalue compound literal in c++?");
  // Defer visiting the literal until the lvalue-to-rvalue conversion. We can
  // only see this when folding in C, so there's no standard to follow here.
  return Success(E);
}

bool LValueExprEvaluator::VisitCXXTypeidExpr(const CXXTypeidExpr *E) {
  if (!E->isPotentiallyEvaluated())
    return Success(E);

  Info.Diag(E, diag::note_constexpr_typeid_polymorphic)
    << E->getExprOperand()->getType()
    << E->getExprOperand()->getSourceRange();
  return false;
}

bool LValueExprEvaluator::VisitCXXUuidofExpr(const CXXUuidofExpr *E) {
  return Success(E);
} 

bool LValueExprEvaluator::VisitMemberExpr(const MemberExpr *E) {
  // Handle static data members.
  if (const VarDecl *VD = dyn_cast<VarDecl>(E->getMemberDecl())) {
    VisitIgnoredValue(E->getBase());
    return VisitVarDecl(E, VD);
  }

  // Handle static member functions.
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(E->getMemberDecl())) {
    if (MD->isStatic()) {
      VisitIgnoredValue(E->getBase());
      return Success(MD);
    }
  }

  // Handle non-static data members.
  return LValueExprEvaluatorBaseTy::VisitMemberExpr(E);
}

bool LValueExprEvaluator::VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
  // FIXME: Deal with vectors as array subscript bases.
  if (E->getBase()->getType()->isVectorType())
    return Error(E);

  if (!EvaluatePointer(E->getBase(), Result, Info))
    return false;

  APSInt Index;
  if (!EvaluateInteger(E->getIdx(), Index, Info))
    return false;
  int64_t IndexValue
    = Index.isSigned() ? Index.getSExtValue()
                       : static_cast<int64_t>(Index.getZExtValue());

  return HandleLValueArrayAdjustment(Info, E, Result, E->getType(), IndexValue);
}

bool LValueExprEvaluator::VisitUnaryDeref(const UnaryOperator *E) {
  return EvaluatePointer(E->getSubExpr(), Result, Info);
}

bool LValueExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (!Visit(E->getSubExpr()))
    return false;
  // __real is a no-op on scalar lvalues.
  if (E->getSubExpr()->getType()->isAnyComplexType())
    HandleLValueComplexElement(Info, E, Result, E->getType(), false);
  return true;
}

bool LValueExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  assert(E->getSubExpr()->getType()->isAnyComplexType() &&
         "lvalue __imag__ on scalar?");
  if (!Visit(E->getSubExpr()))
    return false;
  HandleLValueComplexElement(Info, E, Result, E->getType(), true);
  return true;
}

//===----------------------------------------------------------------------===//
// Pointer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class PointerExprEvaluator
  : public ExprEvaluatorBase<PointerExprEvaluator, bool> {
  LValue &Result;

  bool Success(const Expr *E) {
    Result.set(E);
    return true;
  }
public:

  PointerExprEvaluator(EvalInfo &info, LValue &Result)
    : ExprEvaluatorBaseTy(info), Result(Result) {}

  bool Success(const APValue &V, const Expr *E) {
    Result.setFrom(Info.Ctx, V);
    return true;
  }
  bool ZeroInitialization(const Expr *E) {
    return Success((Expr*)0);
  }

  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitCastExpr(const CastExpr* E);
  bool VisitUnaryAddrOf(const UnaryOperator *E);
  bool VisitObjCStringLiteral(const ObjCStringLiteral *E)
      { return Success(E); }
  bool VisitObjCBoxedExpr(const ObjCBoxedExpr *E)
      { return Success(E); }    
  bool VisitAddrLabelExpr(const AddrLabelExpr *E)
      { return Success(E); }
  bool VisitCallExpr(const CallExpr *E);
  bool VisitBlockExpr(const BlockExpr *E) {
    if (!E->getBlockDecl()->hasCaptures())
      return Success(E);
    return Error(E);
  }
  bool VisitCXXThisExpr(const CXXThisExpr *E) {
    if (!Info.CurrentCall->This)
      return Error(E);
    Result = *Info.CurrentCall->This;
    return true;
  }

  // FIXME: Missing: @protocol, @selector
};
} // end anonymous namespace

static bool EvaluatePointer(const Expr* E, LValue& Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->hasPointerRepresentation());
  return PointerExprEvaluator(Info, Result).Visit(E);
}

bool PointerExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() != BO_Add &&
      E->getOpcode() != BO_Sub)
    return ExprEvaluatorBaseTy::VisitBinaryOperator(E);

  const Expr *PExp = E->getLHS();
  const Expr *IExp = E->getRHS();
  if (IExp->getType()->isPointerType())
    std::swap(PExp, IExp);

  bool EvalPtrOK = EvaluatePointer(PExp, Result, Info);
  if (!EvalPtrOK && !Info.keepEvaluatingAfterFailure())
    return false;

  llvm::APSInt Offset;
  if (!EvaluateInteger(IExp, Offset, Info) || !EvalPtrOK)
    return false;
  int64_t AdditionalOffset
    = Offset.isSigned() ? Offset.getSExtValue()
                        : static_cast<int64_t>(Offset.getZExtValue());
  if (E->getOpcode() == BO_Sub)
    AdditionalOffset = -AdditionalOffset;

  QualType Pointee = PExp->getType()->castAs<PointerType>()->getPointeeType();
  return HandleLValueArrayAdjustment(Info, E, Result, Pointee,
                                     AdditionalOffset);
}

bool PointerExprEvaluator::VisitUnaryAddrOf(const UnaryOperator *E) {
  return EvaluateLValue(E->getSubExpr(), Result, Info);
}

bool PointerExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const Expr* SubExpr = E->getSubExpr();

  switch (E->getCastKind()) {
  default:
    break;

  case CK_BitCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
    if (!Visit(SubExpr))
      return false;
    // Bitcasts to cv void* are static_casts, not reinterpret_casts, so are
    // permitted in constant expressions in C++11. Bitcasts from cv void* are
    // also static_casts, but we disallow them as a resolution to DR1312.
    if (!E->getType()->isVoidPointerType()) {
      Result.Designator.setInvalid();
      if (SubExpr->getType()->isVoidPointerType())
        CCEDiag(E, diag::note_constexpr_invalid_cast)
          << 3 << SubExpr->getType();
      else
        CCEDiag(E, diag::note_constexpr_invalid_cast) << 2;
    }
    return true;

  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase: {
    if (!EvaluatePointer(E->getSubExpr(), Result, Info))
      return false;
    if (!Result.Base && Result.Offset.isZero())
      return true;

    // Now figure out the necessary offset to add to the base LV to get from
    // the derived class to the base class.
    QualType Type =
        E->getSubExpr()->getType()->castAs<PointerType>()->getPointeeType();

    for (CastExpr::path_const_iterator PathI = E->path_begin(),
         PathE = E->path_end(); PathI != PathE; ++PathI) {
      if (!HandleLValueBase(Info, E, Result, Type->getAsCXXRecordDecl(),
                            *PathI))
        return false;
      Type = (*PathI)->getType();
    }

    return true;
  }

  case CK_BaseToDerived:
    if (!Visit(E->getSubExpr()))
      return false;
    if (!Result.Base && Result.Offset.isZero())
      return true;
    return HandleBaseToDerivedCast(Info, E, Result);

  case CK_NullToPointer:
    VisitIgnoredValue(E->getSubExpr());
    return ZeroInitialization(E);

  case CK_IntegralToPointer: {
    CCEDiag(E, diag::note_constexpr_invalid_cast) << 2;

    APValue Value;
    if (!EvaluateIntegerOrLValue(SubExpr, Value, Info))
      break;

    if (Value.isInt()) {
      unsigned Size = Info.Ctx.getTypeSize(E->getType());
      uint64_t N = Value.getInt().extOrTrunc(Size).getZExtValue();
      Result.Base = (Expr*)0;
      Result.Offset = CharUnits::fromQuantity(N);
      Result.CallIndex = 0;
      Result.Designator.setInvalid();
      return true;
    } else {
      // Cast is of an lvalue, no need to change value.
      Result.setFrom(Info.Ctx, Value);
      return true;
    }
  }
  case CK_ArrayToPointerDecay:
    if (SubExpr->isGLValue()) {
      if (!EvaluateLValue(SubExpr, Result, Info))
        return false;
    } else {
      Result.set(SubExpr, Info.CurrentCall->Index);
      if (!EvaluateInPlace(Info.CurrentCall->Temporaries[SubExpr],
                           Info, Result, SubExpr))
        return false;
    }
    // The result is a pointer to the first element of the array.
    if (const ConstantArrayType *CAT
          = Info.Ctx.getAsConstantArrayType(SubExpr->getType()))
      Result.addArray(Info, E, CAT);
    else
      Result.Designator.setInvalid();
    return true;

  case CK_FunctionToPointerDecay:
    return EvaluateLValue(SubExpr, Result, Info);
  }

  return ExprEvaluatorBaseTy::VisitCastExpr(E);
}

bool PointerExprEvaluator::VisitCallExpr(const CallExpr *E) {
  if (IsStringLiteralCall(E))
    return Success(E);

  return ExprEvaluatorBaseTy::VisitCallExpr(E);
}

//===----------------------------------------------------------------------===//
// Member Pointer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class MemberPointerExprEvaluator
  : public ExprEvaluatorBase<MemberPointerExprEvaluator, bool> {
  MemberPtr &Result;

  bool Success(const ValueDecl *D) {
    Result = MemberPtr(D);
    return true;
  }
public:

  MemberPointerExprEvaluator(EvalInfo &Info, MemberPtr &Result)
    : ExprEvaluatorBaseTy(Info), Result(Result) {}

  bool Success(const APValue &V, const Expr *E) {
    Result.setFrom(V);
    return true;
  }
  bool ZeroInitialization(const Expr *E) {
    return Success((const ValueDecl*)0);
  }

  bool VisitCastExpr(const CastExpr *E);
  bool VisitUnaryAddrOf(const UnaryOperator *E);
};
} // end anonymous namespace

static bool EvaluateMemberPointer(const Expr *E, MemberPtr &Result,
                                  EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isMemberPointerType());
  return MemberPointerExprEvaluator(Info, Result).Visit(E);
}

bool MemberPointerExprEvaluator::VisitCastExpr(const CastExpr *E) {
  switch (E->getCastKind()) {
  default:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);

  case CK_NullToMemberPointer:
    VisitIgnoredValue(E->getSubExpr());
    return ZeroInitialization(E);

  case CK_BaseToDerivedMemberPointer: {
    if (!Visit(E->getSubExpr()))
      return false;
    if (E->path_empty())
      return true;
    // Base-to-derived member pointer casts store the path in derived-to-base
    // order, so iterate backwards. The CXXBaseSpecifier also provides us with
    // the wrong end of the derived->base arc, so stagger the path by one class.
    typedef std::reverse_iterator<CastExpr::path_const_iterator> ReverseIter;
    for (ReverseIter PathI(E->path_end() - 1), PathE(E->path_begin());
         PathI != PathE; ++PathI) {
      assert(!(*PathI)->isVirtual() && "memptr cast through vbase");
      const CXXRecordDecl *Derived = (*PathI)->getType()->getAsCXXRecordDecl();
      if (!Result.castToDerived(Derived))
        return Error(E);
    }
    const Type *FinalTy = E->getType()->castAs<MemberPointerType>()->getClass();
    if (!Result.castToDerived(FinalTy->getAsCXXRecordDecl()))
      return Error(E);
    return true;
  }

  case CK_DerivedToBaseMemberPointer:
    if (!Visit(E->getSubExpr()))
      return false;
    for (CastExpr::path_const_iterator PathI = E->path_begin(),
         PathE = E->path_end(); PathI != PathE; ++PathI) {
      assert(!(*PathI)->isVirtual() && "memptr cast through vbase");
      const CXXRecordDecl *Base = (*PathI)->getType()->getAsCXXRecordDecl();
      if (!Result.castToBase(Base))
        return Error(E);
    }
    return true;
  }
}

bool MemberPointerExprEvaluator::VisitUnaryAddrOf(const UnaryOperator *E) {
  // C++11 [expr.unary.op]p3 has very strict rules on how the address of a
  // member can be formed.
  return Success(cast<DeclRefExpr>(E->getSubExpr())->getDecl());
}

//===----------------------------------------------------------------------===//
// Record Evaluation
//===----------------------------------------------------------------------===//

namespace {
  class RecordExprEvaluator
  : public ExprEvaluatorBase<RecordExprEvaluator, bool> {
    const LValue &This;
    APValue &Result;
  public:

    RecordExprEvaluator(EvalInfo &info, const LValue &This, APValue &Result)
      : ExprEvaluatorBaseTy(info), This(This), Result(Result) {}

    bool Success(const APValue &V, const Expr *E) {
      Result = V;
      return true;
    }
    bool ZeroInitialization(const Expr *E);

    bool VisitCastExpr(const CastExpr *E);
    bool VisitInitListExpr(const InitListExpr *E);
    bool VisitCXXConstructExpr(const CXXConstructExpr *E);
  };
}

/// Perform zero-initialization on an object of non-union class type.
/// C++11 [dcl.init]p5:
///  To zero-initialize an object or reference of type T means:
///    [...]
///    -- if T is a (possibly cv-qualified) non-union class type,
///       each non-static data member and each base-class subobject is
///       zero-initialized
static bool HandleClassZeroInitialization(EvalInfo &Info, const Expr *E,
                                          const RecordDecl *RD,
                                          const LValue &This, APValue &Result) {
  assert(!RD->isUnion() && "Expected non-union class type");
  const CXXRecordDecl *CD = dyn_cast<CXXRecordDecl>(RD);
  Result = APValue(APValue::UninitStruct(), CD ? CD->getNumBases() : 0,
                   std::distance(RD->field_begin(), RD->field_end()));

  if (RD->isInvalidDecl()) return false;
  const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(RD);

  if (CD) {
    unsigned Index = 0;
    for (CXXRecordDecl::base_class_const_iterator I = CD->bases_begin(),
           End = CD->bases_end(); I != End; ++I, ++Index) {
      const CXXRecordDecl *Base = I->getType()->getAsCXXRecordDecl();
      LValue Subobject = This;
      if (!HandleLValueDirectBase(Info, E, Subobject, CD, Base, &Layout))
        return false;
      if (!HandleClassZeroInitialization(Info, E, Base, Subobject,
                                         Result.getStructBase(Index)))
        return false;
    }
  }

  for (RecordDecl::field_iterator I = RD->field_begin(), End = RD->field_end();
       I != End; ++I) {
    // -- if T is a reference type, no initialization is performed.
    if (I->getType()->isReferenceType())
      continue;

    LValue Subobject = This;
    if (!HandleLValueMember(Info, E, Subobject, *I, &Layout))
      return false;

    ImplicitValueInitExpr VIE(I->getType());
    if (!EvaluateInPlace(
          Result.getStructField(I->getFieldIndex()), Info, Subobject, &VIE))
      return false;
  }

  return true;
}

bool RecordExprEvaluator::ZeroInitialization(const Expr *E) {
  const RecordDecl *RD = E->getType()->castAs<RecordType>()->getDecl();
  if (RD->isInvalidDecl()) return false;
  if (RD->isUnion()) {
    // C++11 [dcl.init]p5: If T is a (possibly cv-qualified) union type, the
    // object's first non-static named data member is zero-initialized
    RecordDecl::field_iterator I = RD->field_begin();
    if (I == RD->field_end()) {
      Result = APValue((const FieldDecl*)0);
      return true;
    }

    LValue Subobject = This;
    if (!HandleLValueMember(Info, E, Subobject, *I))
      return false;
    Result = APValue(*I);
    ImplicitValueInitExpr VIE(I->getType());
    return EvaluateInPlace(Result.getUnionValue(), Info, Subobject, &VIE);
  }

  if (isa<CXXRecordDecl>(RD) && cast<CXXRecordDecl>(RD)->getNumVBases()) {
    Info.Diag(E, diag::note_constexpr_virtual_base) << RD;
    return false;
  }

  return HandleClassZeroInitialization(Info, E, RD, This, Result);
}

bool RecordExprEvaluator::VisitCastExpr(const CastExpr *E) {
  switch (E->getCastKind()) {
  default:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);

  case CK_ConstructorConversion:
    return Visit(E->getSubExpr());

  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase: {
    APValue DerivedObject;
    if (!Evaluate(DerivedObject, Info, E->getSubExpr()))
      return false;
    if (!DerivedObject.isStruct())
      return Error(E->getSubExpr());

    // Derived-to-base rvalue conversion: just slice off the derived part.
    APValue *Value = &DerivedObject;
    const CXXRecordDecl *RD = E->getSubExpr()->getType()->getAsCXXRecordDecl();
    for (CastExpr::path_const_iterator PathI = E->path_begin(),
         PathE = E->path_end(); PathI != PathE; ++PathI) {
      assert(!(*PathI)->isVirtual() && "record rvalue with virtual base");
      const CXXRecordDecl *Base = (*PathI)->getType()->getAsCXXRecordDecl();
      Value = &Value->getStructBase(getBaseIndex(RD, Base));
      RD = Base;
    }
    Result = *Value;
    return true;
  }
  }
}

bool RecordExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  // Cannot constant-evaluate std::initializer_list inits.
  if (E->initializesStdInitializerList())
    return false;

  const RecordDecl *RD = E->getType()->castAs<RecordType>()->getDecl();
  if (RD->isInvalidDecl()) return false;
  const ASTRecordLayout &Layout = Info.Ctx.getASTRecordLayout(RD);

  if (RD->isUnion()) {
    const FieldDecl *Field = E->getInitializedFieldInUnion();
    Result = APValue(Field);
    if (!Field)
      return true;

    // If the initializer list for a union does not contain any elements, the
    // first element of the union is value-initialized.
    ImplicitValueInitExpr VIE(Field->getType());
    const Expr *InitExpr = E->getNumInits() ? E->getInit(0) : &VIE;

    LValue Subobject = This;
    if (!HandleLValueMember(Info, InitExpr, Subobject, Field, &Layout))
      return false;
    return EvaluateInPlace(Result.getUnionValue(), Info, Subobject, InitExpr);
  }

  assert((!isa<CXXRecordDecl>(RD) || !cast<CXXRecordDecl>(RD)->getNumBases()) &&
         "initializer list for class with base classes");
  Result = APValue(APValue::UninitStruct(), 0,
                   std::distance(RD->field_begin(), RD->field_end()));
  unsigned ElementNo = 0;
  bool Success = true;
  for (RecordDecl::field_iterator Field = RD->field_begin(),
       FieldEnd = RD->field_end(); Field != FieldEnd; ++Field) {
    // Anonymous bit-fields are not considered members of the class for
    // purposes of aggregate initialization.
    if (Field->isUnnamedBitfield())
      continue;

    LValue Subobject = This;

    bool HaveInit = ElementNo < E->getNumInits();

    // FIXME: Diagnostics here should point to the end of the initializer
    // list, not the start.
    if (!HandleLValueMember(Info, HaveInit ? E->getInit(ElementNo) : E,
                            Subobject, *Field, &Layout))
      return false;

    // Perform an implicit value-initialization for members beyond the end of
    // the initializer list.
    ImplicitValueInitExpr VIE(HaveInit ? Info.Ctx.IntTy : Field->getType());

    if (!EvaluateInPlace(
          Result.getStructField(Field->getFieldIndex()),
          Info, Subobject, HaveInit ? E->getInit(ElementNo++) : &VIE)) {
      if (!Info.keepEvaluatingAfterFailure())
        return false;
      Success = false;
    }
  }

  return Success;
}

bool RecordExprEvaluator::VisitCXXConstructExpr(const CXXConstructExpr *E) {
  const CXXConstructorDecl *FD = E->getConstructor();
  if (FD->isInvalidDecl() || FD->getParent()->isInvalidDecl()) return false;

  bool ZeroInit = E->requiresZeroInitialization();
  if (CheckTrivialDefaultConstructor(Info, E->getExprLoc(), FD, ZeroInit)) {
    // If we've already performed zero-initialization, we're already done.
    if (!Result.isUninit())
      return true;

    if (ZeroInit)
      return ZeroInitialization(E);

    const CXXRecordDecl *RD = FD->getParent();
    if (RD->isUnion())
      Result = APValue((FieldDecl*)0);
    else
      Result = APValue(APValue::UninitStruct(), RD->getNumBases(),
                       std::distance(RD->field_begin(), RD->field_end()));
    return true;
  }

  const FunctionDecl *Definition = 0;
  FD->getBody(Definition);

  if (!CheckConstexprFunction(Info, E->getExprLoc(), FD, Definition))
    return false;

  // Avoid materializing a temporary for an elidable copy/move constructor.
  if (E->isElidable() && !ZeroInit)
    if (const MaterializeTemporaryExpr *ME
          = dyn_cast<MaterializeTemporaryExpr>(E->getArg(0)))
      return Visit(ME->GetTemporaryExpr());

  if (ZeroInit && !ZeroInitialization(E))
    return false;

  llvm::ArrayRef<const Expr*> Args(E->getArgs(), E->getNumArgs());
  return HandleConstructorCall(E->getExprLoc(), This, Args,
                               cast<CXXConstructorDecl>(Definition), Info,
                               Result);
}

static bool EvaluateRecord(const Expr *E, const LValue &This,
                           APValue &Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isRecordType() &&
         "can't evaluate expression as a record rvalue");
  return RecordExprEvaluator(Info, This, Result).Visit(E);
}

//===----------------------------------------------------------------------===//
// Temporary Evaluation
//
// Temporaries are represented in the AST as rvalues, but generally behave like
// lvalues. The full-object of which the temporary is a subobject is implicitly
// materialized so that a reference can bind to it.
//===----------------------------------------------------------------------===//
namespace {
class TemporaryExprEvaluator
  : public LValueExprEvaluatorBase<TemporaryExprEvaluator> {
public:
  TemporaryExprEvaluator(EvalInfo &Info, LValue &Result) :
    LValueExprEvaluatorBaseTy(Info, Result) {}

  /// Visit an expression which constructs the value of this temporary.
  bool VisitConstructExpr(const Expr *E) {
    Result.set(E, Info.CurrentCall->Index);
    return EvaluateInPlace(Info.CurrentCall->Temporaries[E], Info, Result, E);
  }

  bool VisitCastExpr(const CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      return LValueExprEvaluatorBaseTy::VisitCastExpr(E);

    case CK_ConstructorConversion:
      return VisitConstructExpr(E->getSubExpr());
    }
  }
  bool VisitInitListExpr(const InitListExpr *E) {
    return VisitConstructExpr(E);
  }
  bool VisitCXXConstructExpr(const CXXConstructExpr *E) {
    return VisitConstructExpr(E);
  }
  bool VisitCallExpr(const CallExpr *E) {
    return VisitConstructExpr(E);
  }
};
} // end anonymous namespace

/// Evaluate an expression of record type as a temporary.
static bool EvaluateTemporary(const Expr *E, LValue &Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isRecordType());
  return TemporaryExprEvaluator(Info, Result).Visit(E);
}

//===----------------------------------------------------------------------===//
// Vector Evaluation
//===----------------------------------------------------------------------===//

namespace {
  class VectorExprEvaluator
  : public ExprEvaluatorBase<VectorExprEvaluator, bool> {
    APValue &Result;
  public:

    VectorExprEvaluator(EvalInfo &info, APValue &Result)
      : ExprEvaluatorBaseTy(info), Result(Result) {}

    bool Success(const ArrayRef<APValue> &V, const Expr *E) {
      assert(V.size() == E->getType()->castAs<VectorType>()->getNumElements());
      // FIXME: remove this APValue copy.
      Result = APValue(V.data(), V.size());
      return true;
    }
    bool Success(const APValue &V, const Expr *E) {
      assert(V.isVector());
      Result = V;
      return true;
    }
    bool ZeroInitialization(const Expr *E);

    bool VisitUnaryReal(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
    bool VisitCastExpr(const CastExpr* E);
    bool VisitInitListExpr(const InitListExpr *E);
    bool VisitUnaryImag(const UnaryOperator *E);
    // FIXME: Missing: unary -, unary ~, binary add/sub/mul/div,
    //                 binary comparisons, binary and/or/xor,
    //                 shufflevector, ExtVectorElementExpr
  };
} // end anonymous namespace

static bool EvaluateVector(const Expr* E, APValue& Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isVectorType() &&"not a vector rvalue");
  return VectorExprEvaluator(Info, Result).Visit(E);
}

bool VectorExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const VectorType *VTy = E->getType()->castAs<VectorType>();
  unsigned NElts = VTy->getNumElements();

  const Expr *SE = E->getSubExpr();
  QualType SETy = SE->getType();

  switch (E->getCastKind()) {
  case CK_VectorSplat: {
    APValue Val = APValue();
    if (SETy->isIntegerType()) {
      APSInt IntResult;
      if (!EvaluateInteger(SE, IntResult, Info))
         return false;
      Val = APValue(IntResult);
    } else if (SETy->isRealFloatingType()) {
       APFloat F(0.0);
       if (!EvaluateFloat(SE, F, Info))
         return false;
       Val = APValue(F);
    } else {
      return Error(E);
    }

    // Splat and create vector APValue.
    SmallVector<APValue, 4> Elts(NElts, Val);
    return Success(Elts, E);
  }
  case CK_BitCast: {
    // Evaluate the operand into an APInt we can extract from.
    llvm::APInt SValInt;
    if (!EvalAndBitcastToAPInt(Info, SE, SValInt))
      return false;
    // Extract the elements
    QualType EltTy = VTy->getElementType();
    unsigned EltSize = Info.Ctx.getTypeSize(EltTy);
    bool BigEndian = Info.Ctx.getTargetInfo().isBigEndian();
    SmallVector<APValue, 4> Elts;
    if (EltTy->isRealFloatingType()) {
      const llvm::fltSemantics &Sem = Info.Ctx.getFloatTypeSemantics(EltTy);
      bool isIEESem = &Sem != &APFloat::PPCDoubleDouble;
      unsigned FloatEltSize = EltSize;
      if (&Sem == &APFloat::x87DoubleExtended)
        FloatEltSize = 80;
      for (unsigned i = 0; i < NElts; i++) {
        llvm::APInt Elt;
        if (BigEndian)
          Elt = SValInt.rotl(i*EltSize+FloatEltSize).trunc(FloatEltSize);
        else
          Elt = SValInt.rotr(i*EltSize).trunc(FloatEltSize);
        Elts.push_back(APValue(APFloat(Elt, isIEESem)));
      }
    } else if (EltTy->isIntegerType()) {
      for (unsigned i = 0; i < NElts; i++) {
        llvm::APInt Elt;
        if (BigEndian)
          Elt = SValInt.rotl(i*EltSize+EltSize).zextOrTrunc(EltSize);
        else
          Elt = SValInt.rotr(i*EltSize).zextOrTrunc(EltSize);
        Elts.push_back(APValue(APSInt(Elt, EltTy->isSignedIntegerType())));
      }
    } else {
      return Error(E);
    }
    return Success(Elts, E);
  }
  default:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);
  }
}

bool
VectorExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  const VectorType *VT = E->getType()->castAs<VectorType>();
  unsigned NumInits = E->getNumInits();
  unsigned NumElements = VT->getNumElements();

  QualType EltTy = VT->getElementType();
  SmallVector<APValue, 4> Elements;

  // The number of initializers can be less than the number of
  // vector elements. For OpenCL, this can be due to nested vector
  // initialization. For GCC compatibility, missing trailing elements 
  // should be initialized with zeroes.
  unsigned CountInits = 0, CountElts = 0;
  while (CountElts < NumElements) {
    // Handle nested vector initialization.
    if (CountInits < NumInits 
        && E->getInit(CountInits)->getType()->isExtVectorType()) {
      APValue v;
      if (!EvaluateVector(E->getInit(CountInits), v, Info))
        return Error(E);
      unsigned vlen = v.getVectorLength();
      for (unsigned j = 0; j < vlen; j++) 
        Elements.push_back(v.getVectorElt(j));
      CountElts += vlen;
    } else if (EltTy->isIntegerType()) {
      llvm::APSInt sInt(32);
      if (CountInits < NumInits) {
        if (!EvaluateInteger(E->getInit(CountInits), sInt, Info))
          return false;
      } else // trailing integer zero.
        sInt = Info.Ctx.MakeIntValue(0, EltTy);
      Elements.push_back(APValue(sInt));
      CountElts++;
    } else {
      llvm::APFloat f(0.0);
      if (CountInits < NumInits) {
        if (!EvaluateFloat(E->getInit(CountInits), f, Info))
          return false;
      } else // trailing float zero.
        f = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(EltTy));
      Elements.push_back(APValue(f));
      CountElts++;
    }
    CountInits++;
  }
  return Success(Elements, E);
}

bool
VectorExprEvaluator::ZeroInitialization(const Expr *E) {
  const VectorType *VT = E->getType()->getAs<VectorType>();
  QualType EltTy = VT->getElementType();
  APValue ZeroElement;
  if (EltTy->isIntegerType())
    ZeroElement = APValue(Info.Ctx.MakeIntValue(0, EltTy));
  else
    ZeroElement =
        APValue(APFloat::getZero(Info.Ctx.getFloatTypeSemantics(EltTy)));

  SmallVector<APValue, 4> Elements(VT->getNumElements(), ZeroElement);
  return Success(Elements, E);
}

bool VectorExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  VisitIgnoredValue(E->getSubExpr());
  return ZeroInitialization(E);
}

//===----------------------------------------------------------------------===//
// Array Evaluation
//===----------------------------------------------------------------------===//

namespace {
  class ArrayExprEvaluator
  : public ExprEvaluatorBase<ArrayExprEvaluator, bool> {
    const LValue &This;
    APValue &Result;
  public:

    ArrayExprEvaluator(EvalInfo &Info, const LValue &This, APValue &Result)
      : ExprEvaluatorBaseTy(Info), This(This), Result(Result) {}

    bool Success(const APValue &V, const Expr *E) {
      assert((V.isArray() || V.isLValue()) &&
             "expected array or string literal");
      Result = V;
      return true;
    }

    bool ZeroInitialization(const Expr *E) {
      const ConstantArrayType *CAT =
          Info.Ctx.getAsConstantArrayType(E->getType());
      if (!CAT)
        return Error(E);

      Result = APValue(APValue::UninitArray(), 0,
                       CAT->getSize().getZExtValue());
      if (!Result.hasArrayFiller()) return true;

      // Zero-initialize all elements.
      LValue Subobject = This;
      Subobject.addArray(Info, E, CAT);
      ImplicitValueInitExpr VIE(CAT->getElementType());
      return EvaluateInPlace(Result.getArrayFiller(), Info, Subobject, &VIE);
    }

    bool VisitInitListExpr(const InitListExpr *E);
    bool VisitCXXConstructExpr(const CXXConstructExpr *E);
  };
} // end anonymous namespace

static bool EvaluateArray(const Expr *E, const LValue &This,
                          APValue &Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isArrayType() && "not an array rvalue");
  return ArrayExprEvaluator(Info, This, Result).Visit(E);
}

bool ArrayExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  const ConstantArrayType *CAT = Info.Ctx.getAsConstantArrayType(E->getType());
  if (!CAT)
    return Error(E);

  // C++11 [dcl.init.string]p1: A char array [...] can be initialized by [...]
  // an appropriately-typed string literal enclosed in braces.
  if (E->isStringLiteralInit()) {
    LValue LV;
    if (!EvaluateLValue(E->getInit(0), LV, Info))
      return false;
    APValue Val;
    LV.moveInto(Val);
    return Success(Val, E);
  }

  bool Success = true;

  assert((!Result.isArray() || Result.getArrayInitializedElts() == 0) &&
         "zero-initialized array shouldn't have any initialized elts");
  APValue Filler;
  if (Result.isArray() && Result.hasArrayFiller())
    Filler = Result.getArrayFiller();

  Result = APValue(APValue::UninitArray(), E->getNumInits(),
                   CAT->getSize().getZExtValue());

  // If the array was previously zero-initialized, preserve the
  // zero-initialized values.
  if (!Filler.isUninit()) {
    for (unsigned I = 0, E = Result.getArrayInitializedElts(); I != E; ++I)
      Result.getArrayInitializedElt(I) = Filler;
    if (Result.hasArrayFiller())
      Result.getArrayFiller() = Filler;
  }

  LValue Subobject = This;
  Subobject.addArray(Info, E, CAT);
  unsigned Index = 0;
  for (InitListExpr::const_iterator I = E->begin(), End = E->end();
       I != End; ++I, ++Index) {
    if (!EvaluateInPlace(Result.getArrayInitializedElt(Index),
                         Info, Subobject, cast<Expr>(*I)) ||
        !HandleLValueArrayAdjustment(Info, cast<Expr>(*I), Subobject,
                                     CAT->getElementType(), 1)) {
      if (!Info.keepEvaluatingAfterFailure())
        return false;
      Success = false;
    }
  }

  if (!Result.hasArrayFiller()) return Success;
  assert(E->hasArrayFiller() && "no array filler for incomplete init list");
  // FIXME: The Subobject here isn't necessarily right. This rarely matters,
  // but sometimes does:
  //   struct S { constexpr S() : p(&p) {} void *p; };
  //   S s[10] = {};
  return EvaluateInPlace(Result.getArrayFiller(), Info,
                         Subobject, E->getArrayFiller()) && Success;
}

bool ArrayExprEvaluator::VisitCXXConstructExpr(const CXXConstructExpr *E) {
  // FIXME: The Subobject here isn't necessarily right. This rarely matters,
  // but sometimes does:
  //   struct S { constexpr S() : p(&p) {} void *p; };
  //   S s[10];
  LValue Subobject = This;

  APValue *Value = &Result;
  bool HadZeroInit = true;
  QualType ElemTy = E->getType();
  while (const ConstantArrayType *CAT =
           Info.Ctx.getAsConstantArrayType(ElemTy)) {
    Subobject.addArray(Info, E, CAT);
    HadZeroInit &= !Value->isUninit();
    if (!HadZeroInit)
      *Value = APValue(APValue::UninitArray(), 0, CAT->getSize().getZExtValue());
    if (!Value->hasArrayFiller())
      return true;
    Value = &Value->getArrayFiller();
    ElemTy = CAT->getElementType();
  }

  if (!ElemTy->isRecordType())
    return Error(E);

  const CXXConstructorDecl *FD = E->getConstructor();

  bool ZeroInit = E->requiresZeroInitialization();
  if (CheckTrivialDefaultConstructor(Info, E->getExprLoc(), FD, ZeroInit)) {
    if (HadZeroInit)
      return true;

    if (ZeroInit) {
      ImplicitValueInitExpr VIE(ElemTy);
      return EvaluateInPlace(*Value, Info, Subobject, &VIE);
    }

    const CXXRecordDecl *RD = FD->getParent();
    if (RD->isUnion())
      *Value = APValue((FieldDecl*)0);
    else
      *Value =
          APValue(APValue::UninitStruct(), RD->getNumBases(),
                  std::distance(RD->field_begin(), RD->field_end()));
    return true;
  }

  const FunctionDecl *Definition = 0;
  FD->getBody(Definition);

  if (!CheckConstexprFunction(Info, E->getExprLoc(), FD, Definition))
    return false;

  if (ZeroInit && !HadZeroInit) {
    ImplicitValueInitExpr VIE(ElemTy);
    if (!EvaluateInPlace(*Value, Info, Subobject, &VIE))
      return false;
  }

  llvm::ArrayRef<const Expr*> Args(E->getArgs(), E->getNumArgs());
  return HandleConstructorCall(E->getExprLoc(), Subobject, Args,
                               cast<CXXConstructorDecl>(Definition),
                               Info, *Value);
}

//===----------------------------------------------------------------------===//
// Integer Evaluation
//
// As a GNU extension, we support casting pointers to sufficiently-wide integer
// types and back in constant folding. Integer values are thus represented
// either as an integer-valued APValue, or as an lvalue-valued APValue.
//===----------------------------------------------------------------------===//

namespace {
class IntExprEvaluator
  : public ExprEvaluatorBase<IntExprEvaluator, bool> {
  APValue &Result;
public:
  IntExprEvaluator(EvalInfo &info, APValue &result)
    : ExprEvaluatorBaseTy(info), Result(result) {}

  bool Success(const llvm::APSInt &SI, const Expr *E, APValue &Result) {
    assert(E->getType()->isIntegralOrEnumerationType() &&
           "Invalid evaluation result.");
    assert(SI.isSigned() == E->getType()->isSignedIntegerOrEnumerationType() &&
           "Invalid evaluation result.");
    assert(SI.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = APValue(SI);
    return true;
  }
  bool Success(const llvm::APSInt &SI, const Expr *E) {
    return Success(SI, E, Result);
  }

  bool Success(const llvm::APInt &I, const Expr *E, APValue &Result) {
    assert(E->getType()->isIntegralOrEnumerationType() && 
           "Invalid evaluation result.");
    assert(I.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = APValue(APSInt(I));
    Result.getInt().setIsUnsigned(
                            E->getType()->isUnsignedIntegerOrEnumerationType());
    return true;
  }
  bool Success(const llvm::APInt &I, const Expr *E) {
    return Success(I, E, Result);
  }

  bool Success(uint64_t Value, const Expr *E, APValue &Result) {
    assert(E->getType()->isIntegralOrEnumerationType() && 
           "Invalid evaluation result.");
    Result = APValue(Info.Ctx.MakeIntValue(Value, E->getType()));
    return true;
  }
  bool Success(uint64_t Value, const Expr *E) {
    return Success(Value, E, Result);
  }

  bool Success(CharUnits Size, const Expr *E) {
    return Success(Size.getQuantity(), E);
  }

  bool Success(const APValue &V, const Expr *E) {
    if (V.isLValue() || V.isAddrLabelDiff()) {
      Result = V;
      return true;
    }
    return Success(V.getInt(), E);
  }

  bool ZeroInitialization(const Expr *E) { return Success(0, E); }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  bool VisitIntegerLiteral(const IntegerLiteral *E) {
    return Success(E->getValue(), E);
  }
  bool VisitCharacterLiteral(const CharacterLiteral *E) {
    return Success(E->getValue(), E);
  }

  bool CheckReferencedDecl(const Expr *E, const Decl *D);
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    if (CheckReferencedDecl(E, E->getDecl()))
      return true;

    return ExprEvaluatorBaseTy::VisitDeclRefExpr(E);
  }
  bool VisitMemberExpr(const MemberExpr *E) {
    if (CheckReferencedDecl(E, E->getMemberDecl())) {
      VisitIgnoredValue(E->getBase());
      return true;
    }

    return ExprEvaluatorBaseTy::VisitMemberExpr(E);
  }

  bool VisitCallExpr(const CallExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitOffsetOfExpr(const OffsetOfExpr *E);
  bool VisitUnaryOperator(const UnaryOperator *E);

  bool VisitCastExpr(const CastExpr* E);
  bool VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E);

  bool VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *E) {
    return Success(E->getValue(), E);
  }
    
  // Note, GNU defines __null as an integer, not a pointer.
  bool VisitGNUNullExpr(const GNUNullExpr *E) {
    return ZeroInitialization(E);
  }

  bool VisitUnaryTypeTraitExpr(const UnaryTypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitBinaryTypeTraitExpr(const BinaryTypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitTypeTraitExpr(const TypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitExpressionTraitExpr(const ExpressionTraitExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

  bool VisitCXXNoexceptExpr(const CXXNoexceptExpr *E);
  bool VisitSizeOfPackExpr(const SizeOfPackExpr *E);

private:
  CharUnits GetAlignOfExpr(const Expr *E);
  CharUnits GetAlignOfType(QualType T);
  static QualType GetObjectType(APValue::LValueBase B);
  bool TryEvaluateBuiltinObjectSize(const CallExpr *E);
  // FIXME: Missing: array subscript of vector, member of vector
};
} // end anonymous namespace

/// EvaluateIntegerOrLValue - Evaluate an rvalue integral-typed expression, and
/// produce either the integer value or a pointer.
///
/// GCC has a heinous extension which folds casts between pointer types and
/// pointer-sized integral types. We support this by allowing the evaluation of
/// an integer rvalue to produce a pointer (represented as an lvalue) instead.
/// Some simple arithmetic on such values is supported (they are treated much
/// like char*).
static bool EvaluateIntegerOrLValue(const Expr *E, APValue &Result,
                                    EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isIntegralOrEnumerationType());
  return IntExprEvaluator(Info, Result).Visit(E);
}

static bool EvaluateInteger(const Expr *E, APSInt &Result, EvalInfo &Info) {
  APValue Val;
  if (!EvaluateIntegerOrLValue(E, Val, Info))
    return false;
  if (!Val.isInt()) {
    // FIXME: It would be better to produce the diagnostic for casting
    //        a pointer to an integer.
    Info.Diag(E, diag::note_invalid_subexpr_in_const_expr);
    return false;
  }
  Result = Val.getInt();
  return true;
}

/// Check whether the given declaration can be directly converted to an integral
/// rvalue. If not, no diagnostic is produced; there are other things we can
/// try.
bool IntExprEvaluator::CheckReferencedDecl(const Expr* E, const Decl* D) {
  // Enums are integer constant exprs.
  if (const EnumConstantDecl *ECD = dyn_cast<EnumConstantDecl>(D)) {
    // Check for signedness/width mismatches between E type and ECD value.
    bool SameSign = (ECD->getInitVal().isSigned()
                     == E->getType()->isSignedIntegerOrEnumerationType());
    bool SameWidth = (ECD->getInitVal().getBitWidth()
                      == Info.Ctx.getIntWidth(E->getType()));
    if (SameSign && SameWidth)
      return Success(ECD->getInitVal(), E);
    else {
      // Get rid of mismatch (otherwise Success assertions will fail)
      // by computing a new value matching the type of E.
      llvm::APSInt Val = ECD->getInitVal();
      if (!SameSign)
        Val.setIsSigned(!ECD->getInitVal().isSigned());
      if (!SameWidth)
        Val = Val.extOrTrunc(Info.Ctx.getIntWidth(E->getType()));
      return Success(Val, E);
    }
  }
  return false;
}

/// EvaluateBuiltinClassifyType - Evaluate __builtin_classify_type the same way
/// as GCC.
static int EvaluateBuiltinClassifyType(const CallExpr *E) {
  // The following enum mimics the values returned by GCC.
  // FIXME: Does GCC differ between lvalue and rvalue references here?
  enum gcc_type_class {
    no_type_class = -1,
    void_type_class, integer_type_class, char_type_class,
    enumeral_type_class, boolean_type_class,
    pointer_type_class, reference_type_class, offset_type_class,
    real_type_class, complex_type_class,
    function_type_class, method_type_class,
    record_type_class, union_type_class,
    array_type_class, string_type_class,
    lang_type_class
  };

  // If no argument was supplied, default to "no_type_class". This isn't
  // ideal, however it is what gcc does.
  if (E->getNumArgs() == 0)
    return no_type_class;

  QualType ArgTy = E->getArg(0)->getType();
  if (ArgTy->isVoidType())
    return void_type_class;
  else if (ArgTy->isEnumeralType())
    return enumeral_type_class;
  else if (ArgTy->isBooleanType())
    return boolean_type_class;
  else if (ArgTy->isCharType())
    return string_type_class; // gcc doesn't appear to use char_type_class
  else if (ArgTy->isIntegerType())
    return integer_type_class;
  else if (ArgTy->isPointerType())
    return pointer_type_class;
  else if (ArgTy->isReferenceType())
    return reference_type_class;
  else if (ArgTy->isRealType())
    return real_type_class;
  else if (ArgTy->isComplexType())
    return complex_type_class;
  else if (ArgTy->isFunctionType())
    return function_type_class;
  else if (ArgTy->isStructureOrClassType())
    return record_type_class;
  else if (ArgTy->isUnionType())
    return union_type_class;
  else if (ArgTy->isArrayType())
    return array_type_class;
  else if (ArgTy->isUnionType())
    return union_type_class;
  else  // FIXME: offset_type_class, method_type_class, & lang_type_class?
    llvm_unreachable("CallExpr::isBuiltinClassifyType(): unimplemented type");
}

/// EvaluateBuiltinConstantPForLValue - Determine the result of
/// __builtin_constant_p when applied to the given lvalue.
///
/// An lvalue is only "constant" if it is a pointer or reference to the first
/// character of a string literal.
template<typename LValue>
static bool EvaluateBuiltinConstantPForLValue(const LValue &LV) {
  const Expr *E = LV.getLValueBase().template dyn_cast<const Expr*>();
  return E && isa<StringLiteral>(E) && LV.getLValueOffset().isZero();
}

/// EvaluateBuiltinConstantP - Evaluate __builtin_constant_p as similarly to
/// GCC as we can manage.
static bool EvaluateBuiltinConstantP(ASTContext &Ctx, const Expr *Arg) {
  QualType ArgType = Arg->getType();

  // __builtin_constant_p always has one operand. The rules which gcc follows
  // are not precisely documented, but are as follows:
  //
  //  - If the operand is of integral, floating, complex or enumeration type,
  //    and can be folded to a known value of that type, it returns 1.
  //  - If the operand and can be folded to a pointer to the first character
  //    of a string literal (or such a pointer cast to an integral type), it
  //    returns 1.
  //
  // Otherwise, it returns 0.
  //
  // FIXME: GCC also intends to return 1 for literals of aggregate types, but
  // its support for this does not currently work.
  if (ArgType->isIntegralOrEnumerationType()) {
    Expr::EvalResult Result;
    if (!Arg->EvaluateAsRValue(Result, Ctx) || Result.HasSideEffects)
      return false;

    APValue &V = Result.Val;
    if (V.getKind() == APValue::Int)
      return true;

    return EvaluateBuiltinConstantPForLValue(V);
  } else if (ArgType->isFloatingType() || ArgType->isAnyComplexType()) {
    return Arg->isEvaluatable(Ctx);
  } else if (ArgType->isPointerType() || Arg->isGLValue()) {
    LValue LV;
    Expr::EvalStatus Status;
    EvalInfo Info(Ctx, Status);
    if ((Arg->isGLValue() ? EvaluateLValue(Arg, LV, Info)
                          : EvaluatePointer(Arg, LV, Info)) &&
        !Status.HasSideEffects)
      return EvaluateBuiltinConstantPForLValue(LV);
  }

  // Anything else isn't considered to be sufficiently constant.
  return false;
}

/// Retrieves the "underlying object type" of the given expression,
/// as used by __builtin_object_size.
QualType IntExprEvaluator::GetObjectType(APValue::LValueBase B) {
  if (const ValueDecl *D = B.dyn_cast<const ValueDecl*>()) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(D))
      return VD->getType();
  } else if (const Expr *E = B.get<const Expr*>()) {
    if (isa<CompoundLiteralExpr>(E))
      return E->getType();
  }

  return QualType();
}

bool IntExprEvaluator::TryEvaluateBuiltinObjectSize(const CallExpr *E) {
  LValue Base;

  {
    // The operand of __builtin_object_size is never evaluated for side-effects.
    // If there are any, but we can determine the pointed-to object anyway, then
    // ignore the side-effects.
    SpeculativeEvaluationRAII SpeculativeEval(Info);
    if (!EvaluatePointer(E->getArg(0), Base, Info))
      return false;
  }

  // If we can prove the base is null, lower to zero now.
  if (!Base.getLValueBase()) return Success(0, E);

  QualType T = GetObjectType(Base.getLValueBase());
  if (T.isNull() ||
      T->isIncompleteType() ||
      T->isFunctionType() ||
      T->isVariablyModifiedType() ||
      T->isDependentType())
    return Error(E);

  CharUnits Size = Info.Ctx.getTypeSizeInChars(T);
  CharUnits Offset = Base.getLValueOffset();

  if (!Offset.isNegative() && Offset <= Size)
    Size -= Offset;
  else
    Size = CharUnits::Zero();
  return Success(Size, E);
}

bool IntExprEvaluator::VisitCallExpr(const CallExpr *E) {
  switch (unsigned BuiltinOp = E->isBuiltinCall()) {
  default:
    return ExprEvaluatorBaseTy::VisitCallExpr(E);

  case Builtin::BI__builtin_object_size: {
    if (TryEvaluateBuiltinObjectSize(E))
      return true;

    // If evaluating the argument has side-effects, we can't determine the size
    // of the object, and so we lower it to unknown now. CodeGen relies on us to
    // handle all cases where the expression has side-effects.
    if (E->getArg(0)->HasSideEffects(Info.Ctx)) {
      if (E->getArg(1)->EvaluateKnownConstInt(Info.Ctx).getZExtValue() <= 1)
        return Success(-1ULL, E);
      return Success(0, E);
    }

    // Expression had no side effects, but we couldn't statically determine the
    // size of the referenced object.
    return Error(E);
  }

  case Builtin::BI__builtin_bswap16:
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64: {
    APSInt Val;
    if (!EvaluateInteger(E->getArg(0), Val, Info))
      return false;

    return Success(Val.byteSwap(), E);
  }

  case Builtin::BI__builtin_classify_type:
    return Success(EvaluateBuiltinClassifyType(E), E);

  case Builtin::BI__builtin_constant_p:
    return Success(EvaluateBuiltinConstantP(Info.Ctx, E->getArg(0)), E);

  case Builtin::BI__builtin_eh_return_data_regno: {
    int Operand = E->getArg(0)->EvaluateKnownConstInt(Info.Ctx).getZExtValue();
    Operand = Info.Ctx.getTargetInfo().getEHDataRegisterNumber(Operand);
    return Success(Operand, E);
  }

  case Builtin::BI__builtin_expect:
    return Visit(E->getArg(0));

  case Builtin::BIstrlen:
    // A call to strlen is not a constant expression.
    if (Info.getLangOpts().CPlusPlus0x)
      Info.CCEDiag(E, diag::note_constexpr_invalid_function)
        << /*isConstexpr*/0 << /*isConstructor*/0 << "'strlen'";
    else
      Info.CCEDiag(E, diag::note_invalid_subexpr_in_const_expr);
    // Fall through.
  case Builtin::BI__builtin_strlen:
    // As an extension, we support strlen() and __builtin_strlen() as constant
    // expressions when the argument is a string literal.
    if (const StringLiteral *S
               = dyn_cast<StringLiteral>(E->getArg(0)->IgnoreParenImpCasts())) {
      // The string literal may have embedded null characters. Find the first
      // one and truncate there.
      StringRef Str = S->getString();
      StringRef::size_type Pos = Str.find(0);
      if (Pos != StringRef::npos)
        Str = Str.substr(0, Pos);
      
      return Success(Str.size(), E);
    }
      
    return Error(E);

  case Builtin::BI__atomic_always_lock_free:
  case Builtin::BI__atomic_is_lock_free:
  case Builtin::BI__c11_atomic_is_lock_free: {
    APSInt SizeVal;
    if (!EvaluateInteger(E->getArg(0), SizeVal, Info))
      return false;

    // For __atomic_is_lock_free(sizeof(_Atomic(T))), if the size is a power
    // of two less than the maximum inline atomic width, we know it is
    // lock-free.  If the size isn't a power of two, or greater than the
    // maximum alignment where we promote atomics, we know it is not lock-free
    // (at least not in the sense of atomic_is_lock_free).  Otherwise,
    // the answer can only be determined at runtime; for example, 16-byte
    // atomics have lock-free implementations on some, but not all,
    // x86-64 processors.

    // Check power-of-two.
    CharUnits Size = CharUnits::fromQuantity(SizeVal.getZExtValue());
    if (Size.isPowerOfTwo()) {
      // Check against inlining width.
      unsigned InlineWidthBits =
          Info.Ctx.getTargetInfo().getMaxAtomicInlineWidth();
      if (Size <= Info.Ctx.toCharUnitsFromBits(InlineWidthBits)) {
        if (BuiltinOp == Builtin::BI__c11_atomic_is_lock_free ||
            Size == CharUnits::One() ||
            E->getArg(1)->isNullPointerConstant(Info.Ctx,
                                                Expr::NPC_NeverValueDependent))
          // OK, we will inline appropriately-aligned operations of this size,
          // and _Atomic(T) is appropriately-aligned.
          return Success(1, E);

        QualType PointeeType = E->getArg(1)->IgnoreImpCasts()->getType()->
          castAs<PointerType>()->getPointeeType();
        if (!PointeeType->isIncompleteType() &&
            Info.Ctx.getTypeAlignInChars(PointeeType) >= Size) {
          // OK, we will inline operations on this object.
          return Success(1, E);
        }
      }
    }

    return BuiltinOp == Builtin::BI__atomic_always_lock_free ?
        Success(0, E) : Error(E);
  }
  }
}

static bool HasSameBase(const LValue &A, const LValue &B) {
  if (!A.getLValueBase())
    return !B.getLValueBase();
  if (!B.getLValueBase())
    return false;

  if (A.getLValueBase().getOpaqueValue() !=
      B.getLValueBase().getOpaqueValue()) {
    const Decl *ADecl = GetLValueBaseDecl(A);
    if (!ADecl)
      return false;
    const Decl *BDecl = GetLValueBaseDecl(B);
    if (!BDecl || ADecl->getCanonicalDecl() != BDecl->getCanonicalDecl())
      return false;
  }

  return IsGlobalLValue(A.getLValueBase()) ||
         A.getLValueCallIndex() == B.getLValueCallIndex();
}

/// Perform the given integer operation, which is known to need at most BitWidth
/// bits, and check for overflow in the original type (if that type was not an
/// unsigned type).
template<typename Operation>
static APSInt CheckedIntArithmetic(EvalInfo &Info, const Expr *E,
                                   const APSInt &LHS, const APSInt &RHS,
                                   unsigned BitWidth, Operation Op) {
  if (LHS.isUnsigned())
    return Op(LHS, RHS);

  APSInt Value(Op(LHS.extend(BitWidth), RHS.extend(BitWidth)), false);
  APSInt Result = Value.trunc(LHS.getBitWidth());
  if (Result.extend(BitWidth) != Value)
    HandleOverflow(Info, E, Value, E->getType());
  return Result;
}

namespace {

/// \brief Data recursive integer evaluator of certain binary operators.
///
/// We use a data recursive algorithm for binary operators so that we are able
/// to handle extreme cases of chained binary operators without causing stack
/// overflow.
class DataRecursiveIntBinOpEvaluator {
  struct EvalResult {
    APValue Val;
    bool Failed;

    EvalResult() : Failed(false) { }

    void swap(EvalResult &RHS) {
      Val.swap(RHS.Val);
      Failed = RHS.Failed;
      RHS.Failed = false;
    }
  };

  struct Job {
    const Expr *E;
    EvalResult LHSResult; // meaningful only for binary operator expression.
    enum { AnyExprKind, BinOpKind, BinOpVisitedLHSKind } Kind;
    
    Job() : StoredInfo(0) { }
    void startSpeculativeEval(EvalInfo &Info) {
      OldEvalStatus = Info.EvalStatus;
      Info.EvalStatus.Diag = 0;
      StoredInfo = &Info;
    }
    ~Job() {
      if (StoredInfo) {
        StoredInfo->EvalStatus = OldEvalStatus;
      }
    }
  private:
    EvalInfo *StoredInfo; // non-null if status changed.
    Expr::EvalStatus OldEvalStatus;
  };

  SmallVector<Job, 16> Queue;

  IntExprEvaluator &IntEval;
  EvalInfo &Info;
  APValue &FinalResult;

public:
  DataRecursiveIntBinOpEvaluator(IntExprEvaluator &IntEval, APValue &Result)
    : IntEval(IntEval), Info(IntEval.getEvalInfo()), FinalResult(Result) { }

  /// \brief True if \param E is a binary operator that we are going to handle
  /// data recursively.
  /// We handle binary operators that are comma, logical, or that have operands
  /// with integral or enumeration type.
  static bool shouldEnqueue(const BinaryOperator *E) {
    return E->getOpcode() == BO_Comma ||
           E->isLogicalOp() ||
           (E->getLHS()->getType()->isIntegralOrEnumerationType() &&
            E->getRHS()->getType()->isIntegralOrEnumerationType());
  }

  bool Traverse(const BinaryOperator *E) {
    enqueue(E);
    EvalResult PrevResult;
    while (!Queue.empty())
      process(PrevResult);

    if (PrevResult.Failed) return false;

    FinalResult.swap(PrevResult.Val);
    return true;
  }

private:
  bool Success(uint64_t Value, const Expr *E, APValue &Result) {
    return IntEval.Success(Value, E, Result);
  }
  bool Success(const APSInt &Value, const Expr *E, APValue &Result) {
    return IntEval.Success(Value, E, Result);
  }
  bool Error(const Expr *E) {
    return IntEval.Error(E);
  }
  bool Error(const Expr *E, diag::kind D) {
    return IntEval.Error(E, D);
  }

  OptionalDiagnostic CCEDiag(const Expr *E, diag::kind D) {
    return Info.CCEDiag(E, D);
  }

  // \brief Returns true if visiting the RHS is necessary, false otherwise.
  bool VisitBinOpLHSOnly(EvalResult &LHSResult, const BinaryOperator *E,
                         bool &SuppressRHSDiags);

  bool VisitBinOp(const EvalResult &LHSResult, const EvalResult &RHSResult,
                  const BinaryOperator *E, APValue &Result);

  void EvaluateExpr(const Expr *E, EvalResult &Result) {
    Result.Failed = !Evaluate(Result.Val, Info, E);
    if (Result.Failed)
      Result.Val = APValue();
  }

  void process(EvalResult &Result);

  void enqueue(const Expr *E) {
    E = E->IgnoreParens();
    Queue.resize(Queue.size()+1);
    Queue.back().E = E;
    Queue.back().Kind = Job::AnyExprKind;
  }
};

}

bool DataRecursiveIntBinOpEvaluator::
       VisitBinOpLHSOnly(EvalResult &LHSResult, const BinaryOperator *E,
                         bool &SuppressRHSDiags) {
  if (E->getOpcode() == BO_Comma) {
    // Ignore LHS but note if we could not evaluate it.
    if (LHSResult.Failed)
      Info.EvalStatus.HasSideEffects = true;
    return true;
  }
  
  if (E->isLogicalOp()) {
    bool lhsResult;
    if (HandleConversionToBool(LHSResult.Val, lhsResult)) {
      // We were able to evaluate the LHS, see if we can get away with not
      // evaluating the RHS: 0 && X -> 0, 1 || X -> 1
      if (lhsResult == (E->getOpcode() == BO_LOr)) {
        Success(lhsResult, E, LHSResult.Val);
        return false; // Ignore RHS
      }
    } else {
      // Since we weren't able to evaluate the left hand side, it
      // must have had side effects.
      Info.EvalStatus.HasSideEffects = true;
      
      // We can't evaluate the LHS; however, sometimes the result
      // is determined by the RHS: X && 0 -> 0, X || 1 -> 1.
      // Don't ignore RHS and suppress diagnostics from this arm.
      SuppressRHSDiags = true;
    }
    
    return true;
  }
  
  assert(E->getLHS()->getType()->isIntegralOrEnumerationType() &&
         E->getRHS()->getType()->isIntegralOrEnumerationType());
  
  if (LHSResult.Failed && !Info.keepEvaluatingAfterFailure())
    return false; // Ignore RHS;

  return true;
}

bool DataRecursiveIntBinOpEvaluator::
       VisitBinOp(const EvalResult &LHSResult, const EvalResult &RHSResult,
                  const BinaryOperator *E, APValue &Result) {
  if (E->getOpcode() == BO_Comma) {
    if (RHSResult.Failed)
      return false;
    Result = RHSResult.Val;
    return true;
  }
  
  if (E->isLogicalOp()) {
    bool lhsResult, rhsResult;
    bool LHSIsOK = HandleConversionToBool(LHSResult.Val, lhsResult);
    bool RHSIsOK = HandleConversionToBool(RHSResult.Val, rhsResult);
    
    if (LHSIsOK) {
      if (RHSIsOK) {
        if (E->getOpcode() == BO_LOr)
          return Success(lhsResult || rhsResult, E, Result);
        else
          return Success(lhsResult && rhsResult, E, Result);
      }
    } else {
      if (RHSIsOK) {
        // We can't evaluate the LHS; however, sometimes the result
        // is determined by the RHS: X && 0 -> 0, X || 1 -> 1.
        if (rhsResult == (E->getOpcode() == BO_LOr))
          return Success(rhsResult, E, Result);
      }
    }
    
    return false;
  }
  
  assert(E->getLHS()->getType()->isIntegralOrEnumerationType() &&
         E->getRHS()->getType()->isIntegralOrEnumerationType());
  
  if (LHSResult.Failed || RHSResult.Failed)
    return false;
  
  const APValue &LHSVal = LHSResult.Val;
  const APValue &RHSVal = RHSResult.Val;
  
  // Handle cases like (unsigned long)&a + 4.
  if (E->isAdditiveOp() && LHSVal.isLValue() && RHSVal.isInt()) {
    Result = LHSVal;
    CharUnits AdditionalOffset = CharUnits::fromQuantity(
                                                         RHSVal.getInt().getZExtValue());
    if (E->getOpcode() == BO_Add)
      Result.getLValueOffset() += AdditionalOffset;
    else
      Result.getLValueOffset() -= AdditionalOffset;
    return true;
  }
  
  // Handle cases like 4 + (unsigned long)&a
  if (E->getOpcode() == BO_Add &&
      RHSVal.isLValue() && LHSVal.isInt()) {
    Result = RHSVal;
    Result.getLValueOffset() += CharUnits::fromQuantity(
                                                        LHSVal.getInt().getZExtValue());
    return true;
  }
  
  if (E->getOpcode() == BO_Sub && LHSVal.isLValue() && RHSVal.isLValue()) {
    // Handle (intptr_t)&&A - (intptr_t)&&B.
    if (!LHSVal.getLValueOffset().isZero() ||
        !RHSVal.getLValueOffset().isZero())
      return false;
    const Expr *LHSExpr = LHSVal.getLValueBase().dyn_cast<const Expr*>();
    const Expr *RHSExpr = RHSVal.getLValueBase().dyn_cast<const Expr*>();
    if (!LHSExpr || !RHSExpr)
      return false;
    const AddrLabelExpr *LHSAddrExpr = dyn_cast<AddrLabelExpr>(LHSExpr);
    const AddrLabelExpr *RHSAddrExpr = dyn_cast<AddrLabelExpr>(RHSExpr);
    if (!LHSAddrExpr || !RHSAddrExpr)
      return false;
    // Make sure both labels come from the same function.
    if (LHSAddrExpr->getLabel()->getDeclContext() !=
        RHSAddrExpr->getLabel()->getDeclContext())
      return false;
    Result = APValue(LHSAddrExpr, RHSAddrExpr);
    return true;
  }
  
  // All the following cases expect both operands to be an integer
  if (!LHSVal.isInt() || !RHSVal.isInt())
    return Error(E);
  
  const APSInt &LHS = LHSVal.getInt();
  APSInt RHS = RHSVal.getInt();
  
  switch (E->getOpcode()) {
    default:
      return Error(E);
    case BO_Mul:
      return Success(CheckedIntArithmetic(Info, E, LHS, RHS,
                                          LHS.getBitWidth() * 2,
                                          std::multiplies<APSInt>()), E,
                     Result);
    case BO_Add:
      return Success(CheckedIntArithmetic(Info, E, LHS, RHS,
                                          LHS.getBitWidth() + 1,
                                          std::plus<APSInt>()), E, Result);
    case BO_Sub:
      return Success(CheckedIntArithmetic(Info, E, LHS, RHS,
                                          LHS.getBitWidth() + 1,
                                          std::minus<APSInt>()), E, Result);
    case BO_And: return Success(LHS & RHS, E, Result);
    case BO_Xor: return Success(LHS ^ RHS, E, Result);
    case BO_Or:  return Success(LHS | RHS, E, Result);
    case BO_Div:
    case BO_Rem:
      if (RHS == 0)
        return Error(E, diag::note_expr_divide_by_zero);
      // Check for overflow case: INT_MIN / -1 or INT_MIN % -1. The latter is
      // not actually undefined behavior in C++11 due to a language defect.
      if (RHS.isNegative() && RHS.isAllOnesValue() &&
          LHS.isSigned() && LHS.isMinSignedValue())
        HandleOverflow(Info, E, -LHS.extend(LHS.getBitWidth() + 1), E->getType());
      return Success(E->getOpcode() == BO_Rem ? LHS % RHS : LHS / RHS, E,
                     Result);
    case BO_Shl: {
      // During constant-folding, a negative shift is an opposite shift. Such
      // a shift is not a constant expression.
      if (RHS.isSigned() && RHS.isNegative()) {
        CCEDiag(E, diag::note_constexpr_negative_shift) << RHS;
        RHS = -RHS;
        goto shift_right;
      }
      
    shift_left:
      // C++11 [expr.shift]p1: Shift width must be less than the bit width of
      // the shifted type.
      unsigned SA = (unsigned) RHS.getLimitedValue(LHS.getBitWidth()-1);
      if (SA != RHS) {
        CCEDiag(E, diag::note_constexpr_large_shift)
        << RHS << E->getType() << LHS.getBitWidth();
      } else if (LHS.isSigned()) {
        // C++11 [expr.shift]p2: A signed left shift must have a non-negative
        // operand, and must not overflow the corresponding unsigned type.
        if (LHS.isNegative())
          CCEDiag(E, diag::note_constexpr_lshift_of_negative) << LHS;
        else if (LHS.countLeadingZeros() < SA)
          CCEDiag(E, diag::note_constexpr_lshift_discards);
      }
      
      return Success(LHS << SA, E, Result);
    }
    case BO_Shr: {
      // During constant-folding, a negative shift is an opposite shift. Such a
      // shift is not a constant expression.
      if (RHS.isSigned() && RHS.isNegative()) {
        CCEDiag(E, diag::note_constexpr_negative_shift) << RHS;
        RHS = -RHS;
        goto shift_left;
      }
      
    shift_right:
      // C++11 [expr.shift]p1: Shift width must be less than the bit width of the
      // shifted type.
      unsigned SA = (unsigned) RHS.getLimitedValue(LHS.getBitWidth()-1);
      if (SA != RHS)
        CCEDiag(E, diag::note_constexpr_large_shift)
        << RHS << E->getType() << LHS.getBitWidth();
      
      return Success(LHS >> SA, E, Result);
    }
      
    case BO_LT: return Success(LHS < RHS, E, Result);
    case BO_GT: return Success(LHS > RHS, E, Result);
    case BO_LE: return Success(LHS <= RHS, E, Result);
    case BO_GE: return Success(LHS >= RHS, E, Result);
    case BO_EQ: return Success(LHS == RHS, E, Result);
    case BO_NE: return Success(LHS != RHS, E, Result);
  }
}

void DataRecursiveIntBinOpEvaluator::process(EvalResult &Result) {
  Job &job = Queue.back();
  
  switch (job.Kind) {
    case Job::AnyExprKind: {
      if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(job.E)) {
        if (shouldEnqueue(Bop)) {
          job.Kind = Job::BinOpKind;
          enqueue(Bop->getLHS());
          return;
        }
      }
      
      EvaluateExpr(job.E, Result);
      Queue.pop_back();
      return;
    }
      
    case Job::BinOpKind: {
      const BinaryOperator *Bop = cast<BinaryOperator>(job.E);
      bool SuppressRHSDiags = false;
      if (!VisitBinOpLHSOnly(Result, Bop, SuppressRHSDiags)) {
        Queue.pop_back();
        return;
      }
      if (SuppressRHSDiags)
        job.startSpeculativeEval(Info);
      job.LHSResult.swap(Result);
      job.Kind = Job::BinOpVisitedLHSKind;
      enqueue(Bop->getRHS());
      return;
    }
      
    case Job::BinOpVisitedLHSKind: {
      const BinaryOperator *Bop = cast<BinaryOperator>(job.E);
      EvalResult RHS;
      RHS.swap(Result);
      Result.Failed = !VisitBinOp(job.LHSResult, RHS, Bop, Result.Val);
      Queue.pop_back();
      return;
    }
  }
  
  llvm_unreachable("Invalid Job::Kind!");
}

bool IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->isAssignmentOp())
    return Error(E);

  if (DataRecursiveIntBinOpEvaluator::shouldEnqueue(E))
    return DataRecursiveIntBinOpEvaluator(*this, Result).Traverse(E);

  QualType LHSTy = E->getLHS()->getType();
  QualType RHSTy = E->getRHS()->getType();

  if (LHSTy->isAnyComplexType()) {
    assert(RHSTy->isAnyComplexType() && "Invalid comparison");
    ComplexValue LHS, RHS;

    bool LHSOK = EvaluateComplex(E->getLHS(), LHS, Info);
    if (!LHSOK && !Info.keepEvaluatingAfterFailure())
      return false;

    if (!EvaluateComplex(E->getRHS(), RHS, Info) || !LHSOK)
      return false;

    if (LHS.isComplexFloat()) {
      APFloat::cmpResult CR_r =
        LHS.getComplexFloatReal().compare(RHS.getComplexFloatReal());
      APFloat::cmpResult CR_i =
        LHS.getComplexFloatImag().compare(RHS.getComplexFloatImag());

      if (E->getOpcode() == BO_EQ)
        return Success((CR_r == APFloat::cmpEqual &&
                        CR_i == APFloat::cmpEqual), E);
      else {
        assert(E->getOpcode() == BO_NE &&
               "Invalid complex comparison.");
        return Success(((CR_r == APFloat::cmpGreaterThan ||
                         CR_r == APFloat::cmpLessThan ||
                         CR_r == APFloat::cmpUnordered) ||
                        (CR_i == APFloat::cmpGreaterThan ||
                         CR_i == APFloat::cmpLessThan ||
                         CR_i == APFloat::cmpUnordered)), E);
      }
    } else {
      if (E->getOpcode() == BO_EQ)
        return Success((LHS.getComplexIntReal() == RHS.getComplexIntReal() &&
                        LHS.getComplexIntImag() == RHS.getComplexIntImag()), E);
      else {
        assert(E->getOpcode() == BO_NE &&
               "Invalid compex comparison.");
        return Success((LHS.getComplexIntReal() != RHS.getComplexIntReal() ||
                        LHS.getComplexIntImag() != RHS.getComplexIntImag()), E);
      }
    }
  }

  if (LHSTy->isRealFloatingType() &&
      RHSTy->isRealFloatingType()) {
    APFloat RHS(0.0), LHS(0.0);

    bool LHSOK = EvaluateFloat(E->getRHS(), RHS, Info);
    if (!LHSOK && !Info.keepEvaluatingAfterFailure())
      return false;

    if (!EvaluateFloat(E->getLHS(), LHS, Info) || !LHSOK)
      return false;

    APFloat::cmpResult CR = LHS.compare(RHS);

    switch (E->getOpcode()) {
    default:
      llvm_unreachable("Invalid binary operator!");
    case BO_LT:
      return Success(CR == APFloat::cmpLessThan, E);
    case BO_GT:
      return Success(CR == APFloat::cmpGreaterThan, E);
    case BO_LE:
      return Success(CR == APFloat::cmpLessThan || CR == APFloat::cmpEqual, E);
    case BO_GE:
      return Success(CR == APFloat::cmpGreaterThan || CR == APFloat::cmpEqual,
                     E);
    case BO_EQ:
      return Success(CR == APFloat::cmpEqual, E);
    case BO_NE:
      return Success(CR == APFloat::cmpGreaterThan
                     || CR == APFloat::cmpLessThan
                     || CR == APFloat::cmpUnordered, E);
    }
  }

  if (LHSTy->isPointerType() && RHSTy->isPointerType()) {
    if (E->getOpcode() == BO_Sub || E->isComparisonOp()) {
      LValue LHSValue, RHSValue;

      bool LHSOK = EvaluatePointer(E->getLHS(), LHSValue, Info);
      if (!LHSOK && Info.keepEvaluatingAfterFailure())
        return false;

      if (!EvaluatePointer(E->getRHS(), RHSValue, Info) || !LHSOK)
        return false;

      // Reject differing bases from the normal codepath; we special-case
      // comparisons to null.
      if (!HasSameBase(LHSValue, RHSValue)) {
        if (E->getOpcode() == BO_Sub) {
          // Handle &&A - &&B.
          if (!LHSValue.Offset.isZero() || !RHSValue.Offset.isZero())
            return false;
          const Expr *LHSExpr = LHSValue.Base.dyn_cast<const Expr*>();
          const Expr *RHSExpr = RHSValue.Base.dyn_cast<const Expr*>();
          if (!LHSExpr || !RHSExpr)
            return false;
          const AddrLabelExpr *LHSAddrExpr = dyn_cast<AddrLabelExpr>(LHSExpr);
          const AddrLabelExpr *RHSAddrExpr = dyn_cast<AddrLabelExpr>(RHSExpr);
          if (!LHSAddrExpr || !RHSAddrExpr)
            return false;
          // Make sure both labels come from the same function.
          if (LHSAddrExpr->getLabel()->getDeclContext() !=
              RHSAddrExpr->getLabel()->getDeclContext())
            return false;
          Result = APValue(LHSAddrExpr, RHSAddrExpr);
          return true;
        }
        // Inequalities and subtractions between unrelated pointers have
        // unspecified or undefined behavior.
        if (!E->isEqualityOp())
          return Error(E);
        // A constant address may compare equal to the address of a symbol.
        // The one exception is that address of an object cannot compare equal
        // to a null pointer constant.
        if ((!LHSValue.Base && !LHSValue.Offset.isZero()) ||
            (!RHSValue.Base && !RHSValue.Offset.isZero()))
          return Error(E);
        // It's implementation-defined whether distinct literals will have
        // distinct addresses. In clang, the result of such a comparison is
        // unspecified, so it is not a constant expression. However, we do know
        // that the address of a literal will be non-null.
        if ((IsLiteralLValue(LHSValue) || IsLiteralLValue(RHSValue)) &&
            LHSValue.Base && RHSValue.Base)
          return Error(E);
        // We can't tell whether weak symbols will end up pointing to the same
        // object.
        if (IsWeakLValue(LHSValue) || IsWeakLValue(RHSValue))
          return Error(E);
        // Pointers with different bases cannot represent the same object.
        // (Note that clang defaults to -fmerge-all-constants, which can
        // lead to inconsistent results for comparisons involving the address
        // of a constant; this generally doesn't matter in practice.)
        return Success(E->getOpcode() == BO_NE, E);
      }

      const CharUnits &LHSOffset = LHSValue.getLValueOffset();
      const CharUnits &RHSOffset = RHSValue.getLValueOffset();

      SubobjectDesignator &LHSDesignator = LHSValue.getLValueDesignator();
      SubobjectDesignator &RHSDesignator = RHSValue.getLValueDesignator();

      if (E->getOpcode() == BO_Sub) {
        // C++11 [expr.add]p6:
        //   Unless both pointers point to elements of the same array object, or
        //   one past the last element of the array object, the behavior is
        //   undefined.
        if (!LHSDesignator.Invalid && !RHSDesignator.Invalid &&
            !AreElementsOfSameArray(getType(LHSValue.Base),
                                    LHSDesignator, RHSDesignator))
          CCEDiag(E, diag::note_constexpr_pointer_subtraction_not_same_array);

        QualType Type = E->getLHS()->getType();
        QualType ElementType = Type->getAs<PointerType>()->getPointeeType();

        CharUnits ElementSize;
        if (!HandleSizeof(Info, E->getExprLoc(), ElementType, ElementSize))
          return false;

        // FIXME: LLVM and GCC both compute LHSOffset - RHSOffset at runtime,
        // and produce incorrect results when it overflows. Such behavior
        // appears to be non-conforming, but is common, so perhaps we should
        // assume the standard intended for such cases to be undefined behavior
        // and check for them.

        // Compute (LHSOffset - RHSOffset) / Size carefully, checking for
        // overflow in the final conversion to ptrdiff_t.
        APSInt LHS(
          llvm::APInt(65, (int64_t)LHSOffset.getQuantity(), true), false);
        APSInt RHS(
          llvm::APInt(65, (int64_t)RHSOffset.getQuantity(), true), false);
        APSInt ElemSize(
          llvm::APInt(65, (int64_t)ElementSize.getQuantity(), true), false);
        APSInt TrueResult = (LHS - RHS) / ElemSize;
        APSInt Result = TrueResult.trunc(Info.Ctx.getIntWidth(E->getType()));

        if (Result.extend(65) != TrueResult)
          HandleOverflow(Info, E, TrueResult, E->getType());
        return Success(Result, E);
      }

      // C++11 [expr.rel]p3:
      //   Pointers to void (after pointer conversions) can be compared, with a
      //   result defined as follows: If both pointers represent the same
      //   address or are both the null pointer value, the result is true if the
      //   operator is <= or >= and false otherwise; otherwise the result is
      //   unspecified.
      // We interpret this as applying to pointers to *cv* void.
      if (LHSTy->isVoidPointerType() && LHSOffset != RHSOffset &&
          E->isRelationalOp())
        CCEDiag(E, diag::note_constexpr_void_comparison);

      // C++11 [expr.rel]p2:
      // - If two pointers point to non-static data members of the same object,
      //   or to subobjects or array elements fo such members, recursively, the
      //   pointer to the later declared member compares greater provided the
      //   two members have the same access control and provided their class is
      //   not a union.
      //   [...]
      // - Otherwise pointer comparisons are unspecified.
      if (!LHSDesignator.Invalid && !RHSDesignator.Invalid &&
          E->isRelationalOp()) {
        bool WasArrayIndex;
        unsigned Mismatch =
          FindDesignatorMismatch(getType(LHSValue.Base), LHSDesignator,
                                 RHSDesignator, WasArrayIndex);
        // At the point where the designators diverge, the comparison has a
        // specified value if:
        //  - we are comparing array indices
        //  - we are comparing fields of a union, or fields with the same access
        // Otherwise, the result is unspecified and thus the comparison is not a
        // constant expression.
        if (!WasArrayIndex && Mismatch < LHSDesignator.Entries.size() &&
            Mismatch < RHSDesignator.Entries.size()) {
          const FieldDecl *LF = getAsField(LHSDesignator.Entries[Mismatch]);
          const FieldDecl *RF = getAsField(RHSDesignator.Entries[Mismatch]);
          if (!LF && !RF)
            CCEDiag(E, diag::note_constexpr_pointer_comparison_base_classes);
          else if (!LF)
            CCEDiag(E, diag::note_constexpr_pointer_comparison_base_field)
              << getAsBaseClass(LHSDesignator.Entries[Mismatch])
              << RF->getParent() << RF;
          else if (!RF)
            CCEDiag(E, diag::note_constexpr_pointer_comparison_base_field)
              << getAsBaseClass(RHSDesignator.Entries[Mismatch])
              << LF->getParent() << LF;
          else if (!LF->getParent()->isUnion() &&
                   LF->getAccess() != RF->getAccess())
            CCEDiag(E, diag::note_constexpr_pointer_comparison_differing_access)
              << LF << LF->getAccess() << RF << RF->getAccess()
              << LF->getParent();
        }
      }

      // The comparison here must be unsigned, and performed with the same
      // width as the pointer.
      unsigned PtrSize = Info.Ctx.getTypeSize(LHSTy);
      uint64_t CompareLHS = LHSOffset.getQuantity();
      uint64_t CompareRHS = RHSOffset.getQuantity();
      assert(PtrSize <= 64 && "Unexpected pointer width");
      uint64_t Mask = ~0ULL >> (64 - PtrSize);
      CompareLHS &= Mask;
      CompareRHS &= Mask;

      // If there is a base and this is a relational operator, we can only
      // compare pointers within the object in question; otherwise, the result
      // depends on where the object is located in memory.
      if (!LHSValue.Base.isNull() && E->isRelationalOp()) {
        QualType BaseTy = getType(LHSValue.Base);
        if (BaseTy->isIncompleteType())
          return Error(E);
        CharUnits Size = Info.Ctx.getTypeSizeInChars(BaseTy);
        uint64_t OffsetLimit = Size.getQuantity();
        if (CompareLHS > OffsetLimit || CompareRHS > OffsetLimit)
          return Error(E);
      }

      switch (E->getOpcode()) {
      default: llvm_unreachable("missing comparison operator");
      case BO_LT: return Success(CompareLHS < CompareRHS, E);
      case BO_GT: return Success(CompareLHS > CompareRHS, E);
      case BO_LE: return Success(CompareLHS <= CompareRHS, E);
      case BO_GE: return Success(CompareLHS >= CompareRHS, E);
      case BO_EQ: return Success(CompareLHS == CompareRHS, E);
      case BO_NE: return Success(CompareLHS != CompareRHS, E);
      }
    }
  }

  if (LHSTy->isMemberPointerType()) {
    assert(E->isEqualityOp() && "unexpected member pointer operation");
    assert(RHSTy->isMemberPointerType() && "invalid comparison");

    MemberPtr LHSValue, RHSValue;

    bool LHSOK = EvaluateMemberPointer(E->getLHS(), LHSValue, Info);
    if (!LHSOK && Info.keepEvaluatingAfterFailure())
      return false;

    if (!EvaluateMemberPointer(E->getRHS(), RHSValue, Info) || !LHSOK)
      return false;

    // C++11 [expr.eq]p2:
    //   If both operands are null, they compare equal. Otherwise if only one is
    //   null, they compare unequal.
    if (!LHSValue.getDecl() || !RHSValue.getDecl()) {
      bool Equal = !LHSValue.getDecl() && !RHSValue.getDecl();
      return Success(E->getOpcode() == BO_EQ ? Equal : !Equal, E);
    }

    //   Otherwise if either is a pointer to a virtual member function, the
    //   result is unspecified.
    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(LHSValue.getDecl()))
      if (MD->isVirtual())
        CCEDiag(E, diag::note_constexpr_compare_virtual_mem_ptr) << MD;
    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(RHSValue.getDecl()))
      if (MD->isVirtual())
        CCEDiag(E, diag::note_constexpr_compare_virtual_mem_ptr) << MD;

    //   Otherwise they compare equal if and only if they would refer to the
    //   same member of the same most derived object or the same subobject if
    //   they were dereferenced with a hypothetical object of the associated
    //   class type.
    bool Equal = LHSValue == RHSValue;
    return Success(E->getOpcode() == BO_EQ ? Equal : !Equal, E);
  }

  if (LHSTy->isNullPtrType()) {
    assert(E->isComparisonOp() && "unexpected nullptr operation");
    assert(RHSTy->isNullPtrType() && "missing pointer conversion");
    // C++11 [expr.rel]p4, [expr.eq]p3: If two operands of type std::nullptr_t
    // are compared, the result is true of the operator is <=, >= or ==, and
    // false otherwise.
    BinaryOperator::Opcode Opcode = E->getOpcode();
    return Success(Opcode == BO_EQ || Opcode == BO_LE || Opcode == BO_GE, E);
  }

  assert((!LHSTy->isIntegralOrEnumerationType() ||
          !RHSTy->isIntegralOrEnumerationType()) &&
         "DataRecursiveIntBinOpEvaluator should have handled integral types");
  // We can't continue from here for non-integral types.
  return ExprEvaluatorBaseTy::VisitBinaryOperator(E);
}

CharUnits IntExprEvaluator::GetAlignOfType(QualType T) {
  // C++ [expr.alignof]p3: "When alignof is applied to a reference type, the
  //   result shall be the alignment of the referenced type."
  if (const ReferenceType *Ref = T->getAs<ReferenceType>())
    T = Ref->getPointeeType();

  // __alignof is defined to return the preferred alignment.
  return Info.Ctx.toCharUnitsFromBits(
    Info.Ctx.getPreferredTypeAlign(T.getTypePtr()));
}

CharUnits IntExprEvaluator::GetAlignOfExpr(const Expr *E) {
  E = E->IgnoreParens();

  // alignof decl is always accepted, even if it doesn't make sense: we default
  // to 1 in those cases.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return Info.Ctx.getDeclAlign(DRE->getDecl(), 
                                 /*RefAsPointee*/true);

  if (const MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return Info.Ctx.getDeclAlign(ME->getMemberDecl(),
                                 /*RefAsPointee*/true);

  return GetAlignOfType(E->getType());
}


/// VisitUnaryExprOrTypeTraitExpr - Evaluate a sizeof, alignof or vec_step with
/// a result as the expression's type.
bool IntExprEvaluator::VisitUnaryExprOrTypeTraitExpr(
                                    const UnaryExprOrTypeTraitExpr *E) {
  switch(E->getKind()) {
  case UETT_AlignOf: {
    if (E->isArgumentType())
      return Success(GetAlignOfType(E->getArgumentType()), E);
    else
      return Success(GetAlignOfExpr(E->getArgumentExpr()), E);
  }

  case UETT_VecStep: {
    QualType Ty = E->getTypeOfArgument();

    if (Ty->isVectorType()) {
      unsigned n = Ty->castAs<VectorType>()->getNumElements();

      // The vec_step built-in functions that take a 3-component
      // vector return 4. (OpenCL 1.1 spec 6.11.12)
      if (n == 3)
        n = 4;

      return Success(n, E);
    } else
      return Success(1, E);
  }

  case UETT_SizeOf: {
    QualType SrcTy = E->getTypeOfArgument();
    // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
    //   the result is the size of the referenced type."
    if (const ReferenceType *Ref = SrcTy->getAs<ReferenceType>())
      SrcTy = Ref->getPointeeType();

    CharUnits Sizeof;
    if (!HandleSizeof(Info, E->getExprLoc(), SrcTy, Sizeof))
      return false;
    return Success(Sizeof, E);
  }
  }

  llvm_unreachable("unknown expr/type trait");
}

bool IntExprEvaluator::VisitOffsetOfExpr(const OffsetOfExpr *OOE) {
  CharUnits Result;
  unsigned n = OOE->getNumComponents();
  if (n == 0)
    return Error(OOE);
  QualType CurrentType = OOE->getTypeSourceInfo()->getType();
  for (unsigned i = 0; i != n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = OOE->getComponent(i);
    switch (ON.getKind()) {
    case OffsetOfExpr::OffsetOfNode::Array: {
      const Expr *Idx = OOE->getIndexExpr(ON.getArrayExprIndex());
      APSInt IdxResult;
      if (!EvaluateInteger(Idx, IdxResult, Info))
        return false;
      const ArrayType *AT = Info.Ctx.getAsArrayType(CurrentType);
      if (!AT)
        return Error(OOE);
      CurrentType = AT->getElementType();
      CharUnits ElementSize = Info.Ctx.getTypeSizeInChars(CurrentType);
      Result += IdxResult.getSExtValue() * ElementSize;
        break;
    }

    case OffsetOfExpr::OffsetOfNode::Field: {
      FieldDecl *MemberDecl = ON.getField();
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT)
        return Error(OOE);
      RecordDecl *RD = RT->getDecl();
      if (RD->isInvalidDecl()) return false;
      const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);
      unsigned i = MemberDecl->getFieldIndex();
      assert(i < RL.getFieldCount() && "offsetof field in wrong type");
      Result += Info.Ctx.toCharUnitsFromBits(RL.getFieldOffset(i));
      CurrentType = MemberDecl->getType().getNonReferenceType();
      break;
    }

    case OffsetOfExpr::OffsetOfNode::Identifier:
      llvm_unreachable("dependent __builtin_offsetof");

    case OffsetOfExpr::OffsetOfNode::Base: {
      CXXBaseSpecifier *BaseSpec = ON.getBase();
      if (BaseSpec->isVirtual())
        return Error(OOE);

      // Find the layout of the class whose base we are looking into.
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT)
        return Error(OOE);
      RecordDecl *RD = RT->getDecl();
      if (RD->isInvalidDecl()) return false;
      const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);

      // Find the base class itself.
      CurrentType = BaseSpec->getType();
      const RecordType *BaseRT = CurrentType->getAs<RecordType>();
      if (!BaseRT)
        return Error(OOE);
      
      // Add the offset to the base.
      Result += RL.getBaseClassOffset(cast<CXXRecordDecl>(BaseRT->getDecl()));
      break;
    }
    }
  }
  return Success(Result, OOE);
}

bool IntExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  switch (E->getOpcode()) {
  default:
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    return Error(E);
  case UO_Extension:
    // FIXME: Should extension allow i-c-e extension expressions in its scope?
    // If so, we could clear the diagnostic ID.
    return Visit(E->getSubExpr());
  case UO_Plus:
    // The result is just the value.
    return Visit(E->getSubExpr());
  case UO_Minus: {
    if (!Visit(E->getSubExpr()))
      return false;
    if (!Result.isInt()) return Error(E);
    const APSInt &Value = Result.getInt();
    if (Value.isSigned() && Value.isMinSignedValue())
      HandleOverflow(Info, E, -Value.extend(Value.getBitWidth() + 1),
                     E->getType());
    return Success(-Value, E);
  }
  case UO_Not: {
    if (!Visit(E->getSubExpr()))
      return false;
    if (!Result.isInt()) return Error(E);
    return Success(~Result.getInt(), E);
  }
  case UO_LNot: {
    bool bres;
    if (!EvaluateAsBooleanCondition(E->getSubExpr(), bres, Info))
      return false;
    return Success(!bres, E);
  }
  }
}

/// HandleCast - This is used to evaluate implicit or explicit casts where the
/// result type is integer.
bool IntExprEvaluator::VisitCastExpr(const CastExpr *E) {
  const Expr *SubExpr = E->getSubExpr();
  QualType DestType = E->getType();
  QualType SrcType = SubExpr->getType();

  switch (E->getCastKind()) {
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_ReinterpretMemberPointer:
  case CK_ConstructorConversion:
  case CK_IntegralToPointer:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralToFloating:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingRealToComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_BuiltinFnToFnPtr:
    llvm_unreachable("invalid cast kind for integral value");

  case CK_BitCast:
  case CK_Dependent:
  case CK_LValueBitCast:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
  case CK_CopyAndAutoreleaseBlockObject:
    return Error(E);

  case CK_UserDefinedConversion:
  case CK_LValueToRValue:
  case CK_AtomicToNonAtomic:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);

  case CK_MemberPointerToBoolean:
  case CK_PointerToBoolean:
  case CK_IntegralToBoolean:
  case CK_FloatingToBoolean:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToBoolean: {
    bool BoolResult;
    if (!EvaluateAsBooleanCondition(SubExpr, BoolResult, Info))
      return false;
    return Success(BoolResult, E);
  }

  case CK_IntegralCast: {
    if (!Visit(SubExpr))
      return false;

    if (!Result.isInt()) {
      // Allow casts of address-of-label differences if they are no-ops
      // or narrowing.  (The narrowing case isn't actually guaranteed to
      // be constant-evaluatable except in some narrow cases which are hard
      // to detect here.  We let it through on the assumption the user knows
      // what they are doing.)
      if (Result.isAddrLabelDiff())
        return Info.Ctx.getTypeSize(DestType) <= Info.Ctx.getTypeSize(SrcType);
      // Only allow casts of lvalues if they are lossless.
      return Info.Ctx.getTypeSize(DestType) == Info.Ctx.getTypeSize(SrcType);
    }

    return Success(HandleIntToIntCast(Info, E, DestType, SrcType,
                                      Result.getInt()), E);
  }

  case CK_PointerToIntegral: {
    CCEDiag(E, diag::note_constexpr_invalid_cast) << 2;

    LValue LV;
    if (!EvaluatePointer(SubExpr, LV, Info))
      return false;

    if (LV.getLValueBase()) {
      // Only allow based lvalue casts if they are lossless.
      // FIXME: Allow a larger integer size than the pointer size, and allow
      // narrowing back down to pointer width in subsequent integral casts.
      // FIXME: Check integer type's active bits, not its type size.
      if (Info.Ctx.getTypeSize(DestType) != Info.Ctx.getTypeSize(SrcType))
        return Error(E);

      LV.Designator.setInvalid();
      LV.moveInto(Result);
      return true;
    }

    APSInt AsInt = Info.Ctx.MakeIntValue(LV.getLValueOffset().getQuantity(), 
                                         SrcType);
    return Success(HandleIntToIntCast(Info, E, DestType, SrcType, AsInt), E);
  }

  case CK_IntegralComplexToReal: {
    ComplexValue C;
    if (!EvaluateComplex(SubExpr, C, Info))
      return false;
    return Success(C.getComplexIntReal(), E);
  }

  case CK_FloatingToIntegral: {
    APFloat F(0.0);
    if (!EvaluateFloat(SubExpr, F, Info))
      return false;

    APSInt Value;
    if (!HandleFloatToIntCast(Info, E, SrcType, F, DestType, Value))
      return false;
    return Success(Value, E);
  }
  }

  llvm_unreachable("unknown cast resulting in integral value");
}

bool IntExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info))
      return false;
    if (!LV.isComplexInt())
      return Error(E);
    return Success(LV.getComplexIntReal(), E);
  }

  return Visit(E->getSubExpr());
}

bool IntExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isComplexIntegerType()) {
    ComplexValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info))
      return false;
    if (!LV.isComplexInt())
      return Error(E);
    return Success(LV.getComplexIntImag(), E);
  }

  VisitIgnoredValue(E->getSubExpr());
  return Success(0, E);
}

bool IntExprEvaluator::VisitSizeOfPackExpr(const SizeOfPackExpr *E) {
  return Success(E->getPackLength(), E);
}

bool IntExprEvaluator::VisitCXXNoexceptExpr(const CXXNoexceptExpr *E) {
  return Success(E->getValue(), E);
}

//===----------------------------------------------------------------------===//
// Float Evaluation
//===----------------------------------------------------------------------===//

namespace {
class FloatExprEvaluator
  : public ExprEvaluatorBase<FloatExprEvaluator, bool> {
  APFloat &Result;
public:
  FloatExprEvaluator(EvalInfo &info, APFloat &result)
    : ExprEvaluatorBaseTy(info), Result(result) {}

  bool Success(const APValue &V, const Expr *e) {
    Result = V.getFloat();
    return true;
  }

  bool ZeroInitialization(const Expr *E) {
    Result = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(E->getType()));
    return true;
  }

  bool VisitCallExpr(const CallExpr *E);

  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitFloatingLiteral(const FloatingLiteral *E);
  bool VisitCastExpr(const CastExpr *E);

  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

  // FIXME: Missing: array subscript of vector, member of vector
};
} // end anonymous namespace

static bool EvaluateFloat(const Expr* E, APFloat& Result, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isRealFloatingType());
  return FloatExprEvaluator(Info, Result).Visit(E);
}

static bool TryEvaluateBuiltinNaN(const ASTContext &Context,
                                  QualType ResultTy,
                                  const Expr *Arg,
                                  bool SNaN,
                                  llvm::APFloat &Result) {
  const StringLiteral *S = dyn_cast<StringLiteral>(Arg->IgnoreParenCasts());
  if (!S) return false;

  const llvm::fltSemantics &Sem = Context.getFloatTypeSemantics(ResultTy);

  llvm::APInt fill;

  // Treat empty strings as if they were zero.
  if (S->getString().empty())
    fill = llvm::APInt(32, 0);
  else if (S->getString().getAsInteger(0, fill))
    return false;

  if (SNaN)
    Result = llvm::APFloat::getSNaN(Sem, false, &fill);
  else
    Result = llvm::APFloat::getQNaN(Sem, false, &fill);
  return true;
}

bool FloatExprEvaluator::VisitCallExpr(const CallExpr *E) {
  switch (E->isBuiltinCall()) {
  default:
    return ExprEvaluatorBaseTy::VisitCallExpr(E);

  case Builtin::BI__builtin_huge_val:
  case Builtin::BI__builtin_huge_valf:
  case Builtin::BI__builtin_huge_vall:
  case Builtin::BI__builtin_inf:
  case Builtin::BI__builtin_inff:
  case Builtin::BI__builtin_infl: {
    const llvm::fltSemantics &Sem =
      Info.Ctx.getFloatTypeSemantics(E->getType());
    Result = llvm::APFloat::getInf(Sem);
    return true;
  }

  case Builtin::BI__builtin_nans:
  case Builtin::BI__builtin_nansf:
  case Builtin::BI__builtin_nansl:
    if (!TryEvaluateBuiltinNaN(Info.Ctx, E->getType(), E->getArg(0),
                               true, Result))
      return Error(E);
    return true;

  case Builtin::BI__builtin_nan:
  case Builtin::BI__builtin_nanf:
  case Builtin::BI__builtin_nanl:
    // If this is __builtin_nan() turn this into a nan, otherwise we
    // can't constant fold it.
    if (!TryEvaluateBuiltinNaN(Info.Ctx, E->getType(), E->getArg(0),
                               false, Result))
      return Error(E);
    return true;

  case Builtin::BI__builtin_fabs:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BI__builtin_fabsl:
    if (!EvaluateFloat(E->getArg(0), Result, Info))
      return false;

    if (Result.isNegative())
      Result.changeSign();
    return true;

  case Builtin::BI__builtin_copysign:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BI__builtin_copysignl: {
    APFloat RHS(0.);
    if (!EvaluateFloat(E->getArg(0), Result, Info) ||
        !EvaluateFloat(E->getArg(1), RHS, Info))
      return false;
    Result.copySign(RHS);
    return true;
  }
  }
}

bool FloatExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue CV;
    if (!EvaluateComplex(E->getSubExpr(), CV, Info))
      return false;
    Result = CV.FloatReal;
    return true;
  }

  return Visit(E->getSubExpr());
}

bool FloatExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue CV;
    if (!EvaluateComplex(E->getSubExpr(), CV, Info))
      return false;
    Result = CV.FloatImag;
    return true;
  }

  VisitIgnoredValue(E->getSubExpr());
  const llvm::fltSemantics &Sem = Info.Ctx.getFloatTypeSemantics(E->getType());
  Result = llvm::APFloat::getZero(Sem);
  return true;
}

bool FloatExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  switch (E->getOpcode()) {
  default: return Error(E);
  case UO_Plus:
    return EvaluateFloat(E->getSubExpr(), Result, Info);
  case UO_Minus:
    if (!EvaluateFloat(E->getSubExpr(), Result, Info))
      return false;
    Result.changeSign();
    return true;
  }
}

bool FloatExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->isPtrMemOp() || E->isAssignmentOp() || E->getOpcode() == BO_Comma)
    return ExprEvaluatorBaseTy::VisitBinaryOperator(E);

  APFloat RHS(0.0);
  bool LHSOK = EvaluateFloat(E->getLHS(), Result, Info);
  if (!LHSOK && !Info.keepEvaluatingAfterFailure())
    return false;
  if (!EvaluateFloat(E->getRHS(), RHS, Info) || !LHSOK)
    return false;

  switch (E->getOpcode()) {
  default: return Error(E);
  case BO_Mul:
    Result.multiply(RHS, APFloat::rmNearestTiesToEven);
    break;
  case BO_Add:
    Result.add(RHS, APFloat::rmNearestTiesToEven);
    break;
  case BO_Sub:
    Result.subtract(RHS, APFloat::rmNearestTiesToEven);
    break;
  case BO_Div:
    Result.divide(RHS, APFloat::rmNearestTiesToEven);
    break;
  }

  if (Result.isInfinity() || Result.isNaN())
    CCEDiag(E, diag::note_constexpr_float_arithmetic) << Result.isNaN();
  return true;
}

bool FloatExprEvaluator::VisitFloatingLiteral(const FloatingLiteral *E) {
  Result = E->getValue();
  return true;
}

bool FloatExprEvaluator::VisitCastExpr(const CastExpr *E) {
  const Expr* SubExpr = E->getSubExpr();

  switch (E->getCastKind()) {
  default:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);

  case CK_IntegralToFloating: {
    APSInt IntResult;
    return EvaluateInteger(SubExpr, IntResult, Info) &&
           HandleIntToFloatCast(Info, E, SubExpr->getType(), IntResult,
                                E->getType(), Result);
  }

  case CK_FloatingCast: {
    if (!Visit(SubExpr))
      return false;
    return HandleFloatToFloatCast(Info, E, SubExpr->getType(), E->getType(),
                                  Result);
  }

  case CK_FloatingComplexToReal: {
    ComplexValue V;
    if (!EvaluateComplex(SubExpr, V, Info))
      return false;
    Result = V.getComplexFloatReal();
    return true;
  }
  }
}

//===----------------------------------------------------------------------===//
// Complex Evaluation (for float and integer)
//===----------------------------------------------------------------------===//

namespace {
class ComplexExprEvaluator
  : public ExprEvaluatorBase<ComplexExprEvaluator, bool> {
  ComplexValue &Result;

public:
  ComplexExprEvaluator(EvalInfo &info, ComplexValue &Result)
    : ExprEvaluatorBaseTy(info), Result(Result) {}

  bool Success(const APValue &V, const Expr *e) {
    Result.setFrom(V);
    return true;
  }

  bool ZeroInitialization(const Expr *E);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  bool VisitImaginaryLiteral(const ImaginaryLiteral *E);
  bool VisitCastExpr(const CastExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitInitListExpr(const InitListExpr *E);
};
} // end anonymous namespace

static bool EvaluateComplex(const Expr *E, ComplexValue &Result,
                            EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isAnyComplexType());
  return ComplexExprEvaluator(Info, Result).Visit(E);
}

bool ComplexExprEvaluator::ZeroInitialization(const Expr *E) {
  QualType ElemTy = E->getType()->castAs<ComplexType>()->getElementType();
  if (ElemTy->isRealFloatingType()) {
    Result.makeComplexFloat();
    APFloat Zero = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(ElemTy));
    Result.FloatReal = Zero;
    Result.FloatImag = Zero;
  } else {
    Result.makeComplexInt();
    APSInt Zero = Info.Ctx.MakeIntValue(0, ElemTy);
    Result.IntReal = Zero;
    Result.IntImag = Zero;
  }
  return true;
}

bool ComplexExprEvaluator::VisitImaginaryLiteral(const ImaginaryLiteral *E) {
  const Expr* SubExpr = E->getSubExpr();

  if (SubExpr->getType()->isRealFloatingType()) {
    Result.makeComplexFloat();
    APFloat &Imag = Result.FloatImag;
    if (!EvaluateFloat(SubExpr, Imag, Info))
      return false;

    Result.FloatReal = APFloat(Imag.getSemantics());
    return true;
  } else {
    assert(SubExpr->getType()->isIntegerType() &&
           "Unexpected imaginary literal.");

    Result.makeComplexInt();
    APSInt &Imag = Result.IntImag;
    if (!EvaluateInteger(SubExpr, Imag, Info))
      return false;

    Result.IntReal = APSInt(Imag.getBitWidth(), !Imag.isSigned());
    return true;
  }
}

bool ComplexExprEvaluator::VisitCastExpr(const CastExpr *E) {

  switch (E->getCastKind()) {
  case CK_BitCast:
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ReinterpretMemberPointer:
  case CK_ConstructorConversion:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_BuiltinFnToFnPtr:
    llvm_unreachable("invalid cast kind for complex value");

  case CK_LValueToRValue:
  case CK_AtomicToNonAtomic:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
    return ExprEvaluatorBaseTy::VisitCastExpr(E);

  case CK_Dependent:
  case CK_LValueBitCast:
  case CK_UserDefinedConversion:
    return Error(E);

  case CK_FloatingRealToComplex: {
    APFloat &Real = Result.FloatReal;
    if (!EvaluateFloat(E->getSubExpr(), Real, Info))
      return false;

    Result.makeComplexFloat();
    Result.FloatImag = APFloat(Real.getSemantics());
    return true;
  }

  case CK_FloatingComplexCast: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();

    return HandleFloatToFloatCast(Info, E, From, To, Result.FloatReal) &&
           HandleFloatToFloatCast(Info, E, From, To, Result.FloatImag);
  }

  case CK_FloatingComplexToIntegralComplex: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();
    Result.makeComplexInt();
    return HandleFloatToIntCast(Info, E, From, Result.FloatReal,
                                To, Result.IntReal) &&
           HandleFloatToIntCast(Info, E, From, Result.FloatImag,
                                To, Result.IntImag);
  }

  case CK_IntegralRealToComplex: {
    APSInt &Real = Result.IntReal;
    if (!EvaluateInteger(E->getSubExpr(), Real, Info))
      return false;

    Result.makeComplexInt();
    Result.IntImag = APSInt(Real.getBitWidth(), !Real.isSigned());
    return true;
  }

  case CK_IntegralComplexCast: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->getAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->getAs<ComplexType>()->getElementType();

    Result.IntReal = HandleIntToIntCast(Info, E, To, From, Result.IntReal);
    Result.IntImag = HandleIntToIntCast(Info, E, To, From, Result.IntImag);
    return true;
  }

  case CK_IntegralComplexToFloatingComplex: {
    if (!Visit(E->getSubExpr()))
      return false;

    QualType To = E->getType()->castAs<ComplexType>()->getElementType();
    QualType From
      = E->getSubExpr()->getType()->castAs<ComplexType>()->getElementType();
    Result.makeComplexFloat();
    return HandleIntToFloatCast(Info, E, From, Result.IntReal,
                                To, Result.FloatReal) &&
           HandleIntToFloatCast(Info, E, From, Result.IntImag,
                                To, Result.FloatImag);
  }
  }

  llvm_unreachable("unknown cast resulting in complex value");
}

bool ComplexExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->isPtrMemOp() || E->isAssignmentOp() || E->getOpcode() == BO_Comma)
    return ExprEvaluatorBaseTy::VisitBinaryOperator(E);

  bool LHSOK = Visit(E->getLHS());
  if (!LHSOK && !Info.keepEvaluatingAfterFailure())
    return false;

  ComplexValue RHS;
  if (!EvaluateComplex(E->getRHS(), RHS, Info) || !LHSOK)
    return false;

  assert(Result.isComplexFloat() == RHS.isComplexFloat() &&
         "Invalid operands to binary operator.");
  switch (E->getOpcode()) {
  default: return Error(E);
  case BO_Add:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().add(RHS.getComplexFloatReal(),
                                       APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().add(RHS.getComplexFloatImag(),
                                       APFloat::rmNearestTiesToEven);
    } else {
      Result.getComplexIntReal() += RHS.getComplexIntReal();
      Result.getComplexIntImag() += RHS.getComplexIntImag();
    }
    break;
  case BO_Sub:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().subtract(RHS.getComplexFloatReal(),
                                            APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().subtract(RHS.getComplexFloatImag(),
                                            APFloat::rmNearestTiesToEven);
    } else {
      Result.getComplexIntReal() -= RHS.getComplexIntReal();
      Result.getComplexIntImag() -= RHS.getComplexIntImag();
    }
    break;
  case BO_Mul:
    if (Result.isComplexFloat()) {
      ComplexValue LHS = Result;
      APFloat &LHS_r = LHS.getComplexFloatReal();
      APFloat &LHS_i = LHS.getComplexFloatImag();
      APFloat &RHS_r = RHS.getComplexFloatReal();
      APFloat &RHS_i = RHS.getComplexFloatImag();

      APFloat Tmp = LHS_r;
      Tmp.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatReal() = Tmp;
      Tmp = LHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatReal().subtract(Tmp, APFloat::rmNearestTiesToEven);

      Tmp = LHS_r;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag() = Tmp;
      Tmp = LHS_i;
      Tmp.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().add(Tmp, APFloat::rmNearestTiesToEven);
    } else {
      ComplexValue LHS = Result;
      Result.getComplexIntReal() =
        (LHS.getComplexIntReal() * RHS.getComplexIntReal() -
         LHS.getComplexIntImag() * RHS.getComplexIntImag());
      Result.getComplexIntImag() =
        (LHS.getComplexIntReal() * RHS.getComplexIntImag() +
         LHS.getComplexIntImag() * RHS.getComplexIntReal());
    }
    break;
  case BO_Div:
    if (Result.isComplexFloat()) {
      ComplexValue LHS = Result;
      APFloat &LHS_r = LHS.getComplexFloatReal();
      APFloat &LHS_i = LHS.getComplexFloatImag();
      APFloat &RHS_r = RHS.getComplexFloatReal();
      APFloat &RHS_i = RHS.getComplexFloatImag();
      APFloat &Res_r = Result.getComplexFloatReal();
      APFloat &Res_i = Result.getComplexFloatImag();

      APFloat Den = RHS_r;
      Den.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      APFloat Tmp = RHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Den.add(Tmp, APFloat::rmNearestTiesToEven);

      Res_r = LHS_r;
      Res_r.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Tmp = LHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Res_r.add(Tmp, APFloat::rmNearestTiesToEven);
      Res_r.divide(Den, APFloat::rmNearestTiesToEven);

      Res_i = LHS_i;
      Res_i.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Tmp = LHS_r;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Res_i.subtract(Tmp, APFloat::rmNearestTiesToEven);
      Res_i.divide(Den, APFloat::rmNearestTiesToEven);
    } else {
      if (RHS.getComplexIntReal() == 0 && RHS.getComplexIntImag() == 0)
        return Error(E, diag::note_expr_divide_by_zero);

      ComplexValue LHS = Result;
      APSInt Den = RHS.getComplexIntReal() * RHS.getComplexIntReal() +
        RHS.getComplexIntImag() * RHS.getComplexIntImag();
      Result.getComplexIntReal() =
        (LHS.getComplexIntReal() * RHS.getComplexIntReal() +
         LHS.getComplexIntImag() * RHS.getComplexIntImag()) / Den;
      Result.getComplexIntImag() =
        (LHS.getComplexIntImag() * RHS.getComplexIntReal() -
         LHS.getComplexIntReal() * RHS.getComplexIntImag()) / Den;
    }
    break;
  }

  return true;
}

bool ComplexExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  // Get the operand value into 'Result'.
  if (!Visit(E->getSubExpr()))
    return false;

  switch (E->getOpcode()) {
  default:
    return Error(E);
  case UO_Extension:
    return true;
  case UO_Plus:
    // The result is always just the subexpr.
    return true;
  case UO_Minus:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().changeSign();
      Result.getComplexFloatImag().changeSign();
    }
    else {
      Result.getComplexIntReal() = -Result.getComplexIntReal();
      Result.getComplexIntImag() = -Result.getComplexIntImag();
    }
    return true;
  case UO_Not:
    if (Result.isComplexFloat())
      Result.getComplexFloatImag().changeSign();
    else
      Result.getComplexIntImag() = -Result.getComplexIntImag();
    return true;
  }
}

bool ComplexExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  if (E->getNumInits() == 2) {
    if (E->getType()->isComplexType()) {
      Result.makeComplexFloat();
      if (!EvaluateFloat(E->getInit(0), Result.FloatReal, Info))
        return false;
      if (!EvaluateFloat(E->getInit(1), Result.FloatImag, Info))
        return false;
    } else {
      Result.makeComplexInt();
      if (!EvaluateInteger(E->getInit(0), Result.IntReal, Info))
        return false;
      if (!EvaluateInteger(E->getInit(1), Result.IntImag, Info))
        return false;
    }
    return true;
  }
  return ExprEvaluatorBaseTy::VisitInitListExpr(E);
}

//===----------------------------------------------------------------------===//
// Void expression evaluation, primarily for a cast to void on the LHS of a
// comma operator
//===----------------------------------------------------------------------===//

namespace {
class VoidExprEvaluator
  : public ExprEvaluatorBase<VoidExprEvaluator, bool> {
public:
  VoidExprEvaluator(EvalInfo &Info) : ExprEvaluatorBaseTy(Info) {}

  bool Success(const APValue &V, const Expr *e) { return true; }

  bool VisitCastExpr(const CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      return ExprEvaluatorBaseTy::VisitCastExpr(E);
    case CK_ToVoid:
      VisitIgnoredValue(E->getSubExpr());
      return true;
    }
  }
};
} // end anonymous namespace

static bool EvaluateVoid(const Expr *E, EvalInfo &Info) {
  assert(E->isRValue() && E->getType()->isVoidType());
  return VoidExprEvaluator(Info).Visit(E);
}

//===----------------------------------------------------------------------===//
// Top level Expr::EvaluateAsRValue method.
//===----------------------------------------------------------------------===//

static bool Evaluate(APValue &Result, EvalInfo &Info, const Expr *E) {
  // In C, function designators are not lvalues, but we evaluate them as if they
  // are.
  if (E->isGLValue() || E->getType()->isFunctionType()) {
    LValue LV;
    if (!EvaluateLValue(E, LV, Info))
      return false;
    LV.moveInto(Result);
  } else if (E->getType()->isVectorType()) {
    if (!EvaluateVector(E, Result, Info))
      return false;
  } else if (E->getType()->isIntegralOrEnumerationType()) {
    if (!IntExprEvaluator(Info, Result).Visit(E))
      return false;
  } else if (E->getType()->hasPointerRepresentation()) {
    LValue LV;
    if (!EvaluatePointer(E, LV, Info))
      return false;
    LV.moveInto(Result);
  } else if (E->getType()->isRealFloatingType()) {
    llvm::APFloat F(0.0);
    if (!EvaluateFloat(E, F, Info))
      return false;
    Result = APValue(F);
  } else if (E->getType()->isAnyComplexType()) {
    ComplexValue C;
    if (!EvaluateComplex(E, C, Info))
      return false;
    C.moveInto(Result);
  } else if (E->getType()->isMemberPointerType()) {
    MemberPtr P;
    if (!EvaluateMemberPointer(E, P, Info))
      return false;
    P.moveInto(Result);
    return true;
  } else if (E->getType()->isArrayType()) {
    LValue LV;
    LV.set(E, Info.CurrentCall->Index);
    if (!EvaluateArray(E, LV, Info.CurrentCall->Temporaries[E], Info))
      return false;
    Result = Info.CurrentCall->Temporaries[E];
  } else if (E->getType()->isRecordType()) {
    LValue LV;
    LV.set(E, Info.CurrentCall->Index);
    if (!EvaluateRecord(E, LV, Info.CurrentCall->Temporaries[E], Info))
      return false;
    Result = Info.CurrentCall->Temporaries[E];
  } else if (E->getType()->isVoidType()) {
    if (!Info.getLangOpts().CPlusPlus0x)
      Info.CCEDiag(E, diag::note_constexpr_nonliteral)
        << E->getType();
    if (!EvaluateVoid(E, Info))
      return false;
  } else if (Info.getLangOpts().CPlusPlus0x) {
    Info.Diag(E, diag::note_constexpr_nonliteral) << E->getType();
    return false;
  } else {
    Info.Diag(E, diag::note_invalid_subexpr_in_const_expr);
    return false;
  }

  return true;
}

/// EvaluateInPlace - Evaluate an expression in-place in an APValue. In some
/// cases, the in-place evaluation is essential, since later initializers for
/// an object can indirectly refer to subobjects which were initialized earlier.
static bool EvaluateInPlace(APValue &Result, EvalInfo &Info, const LValue &This,
                            const Expr *E, CheckConstantExpressionKind CCEK,
                            bool AllowNonLiteralTypes) {
  if (!AllowNonLiteralTypes && !CheckLiteralType(Info, E))
    return false;

  if (E->isRValue()) {
    // Evaluate arrays and record types in-place, so that later initializers can
    // refer to earlier-initialized members of the object.
    if (E->getType()->isArrayType())
      return EvaluateArray(E, This, Result, Info);
    else if (E->getType()->isRecordType())
      return EvaluateRecord(E, This, Result, Info);
  }

  // For any other type, in-place evaluation is unimportant.
  return Evaluate(Result, Info, E);
}

/// EvaluateAsRValue - Try to evaluate this expression, performing an implicit
/// lvalue-to-rvalue cast if it is an lvalue.
static bool EvaluateAsRValue(EvalInfo &Info, const Expr *E, APValue &Result) {
  if (!CheckLiteralType(Info, E))
    return false;

  if (!::Evaluate(Result, Info, E))
    return false;

  if (E->isGLValue()) {
    LValue LV;
    LV.setFrom(Info.Ctx, Result);
    if (!HandleLValueToRValueConversion(Info, E, E->getType(), LV, Result))
      return false;
  }

  // Check this core constant expression is a constant expression.
  return CheckConstantExpression(Info, E->getExprLoc(), E->getType(), Result);
}

/// EvaluateAsRValue - Return true if this is a constant which we can fold using
/// any crazy technique (that has nothing to do with language standards) that
/// we want to.  If this function returns true, it returns the folded constant
/// in Result. If this expression is a glvalue, an lvalue-to-rvalue conversion
/// will be applied to the result.
bool Expr::EvaluateAsRValue(EvalResult &Result, const ASTContext &Ctx) const {
  // Fast-path evaluations of integer literals, since we sometimes see files
  // containing vast quantities of these.
  if (const IntegerLiteral *L = dyn_cast<IntegerLiteral>(this)) {
    Result.Val = APValue(APSInt(L->getValue(),
                                L->getType()->isUnsignedIntegerType()));
    return true;
  }

  // FIXME: Evaluating values of large array and record types can cause
  // performance problems. Only do so in C++11 for now.
  if (isRValue() && (getType()->isArrayType() || getType()->isRecordType()) &&
      !Ctx.getLangOpts().CPlusPlus0x)
    return false;

  EvalInfo Info(Ctx, Result);
  return ::EvaluateAsRValue(Info, this, Result.Val);
}

bool Expr::EvaluateAsBooleanCondition(bool &Result,
                                      const ASTContext &Ctx) const {
  EvalResult Scratch;
  return EvaluateAsRValue(Scratch, Ctx) &&
         HandleConversionToBool(Scratch.Val, Result);
}

bool Expr::EvaluateAsInt(APSInt &Result, const ASTContext &Ctx,
                         SideEffectsKind AllowSideEffects) const {
  if (!getType()->isIntegralOrEnumerationType())
    return false;

  EvalResult ExprResult;
  if (!EvaluateAsRValue(ExprResult, Ctx) || !ExprResult.Val.isInt() ||
      (!AllowSideEffects && ExprResult.HasSideEffects))
    return false;

  Result = ExprResult.Val.getInt();
  return true;
}

bool Expr::EvaluateAsLValue(EvalResult &Result, const ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result);

  LValue LV;
  if (!EvaluateLValue(this, LV, Info) || Result.HasSideEffects ||
      !CheckLValueConstantExpression(Info, getExprLoc(),
                                     Ctx.getLValueReferenceType(getType()), LV))
    return false;

  LV.moveInto(Result.Val);
  return true;
}

bool Expr::EvaluateAsInitializer(APValue &Value, const ASTContext &Ctx,
                                 const VarDecl *VD,
                      llvm::SmallVectorImpl<PartialDiagnosticAt> &Notes) const {
  // FIXME: Evaluating initializers for large array and record types can cause
  // performance problems. Only do so in C++11 for now.
  if (isRValue() && (getType()->isArrayType() || getType()->isRecordType()) &&
      !Ctx.getLangOpts().CPlusPlus0x)
    return false;

  Expr::EvalStatus EStatus;
  EStatus.Diag = &Notes;

  EvalInfo InitInfo(Ctx, EStatus);
  InitInfo.setEvaluatingDecl(VD, Value);

  LValue LVal;
  LVal.set(VD);

  // C++11 [basic.start.init]p2:
  //  Variables with static storage duration or thread storage duration shall be
  //  zero-initialized before any other initialization takes place.
  // This behavior is not present in C.
  if (Ctx.getLangOpts().CPlusPlus && !VD->hasLocalStorage() &&
      !VD->getType()->isReferenceType()) {
    ImplicitValueInitExpr VIE(VD->getType());
    if (!EvaluateInPlace(Value, InitInfo, LVal, &VIE, CCEK_Constant,
                         /*AllowNonLiteralTypes=*/true))
      return false;
  }

  if (!EvaluateInPlace(Value, InitInfo, LVal, this, CCEK_Constant,
                         /*AllowNonLiteralTypes=*/true) ||
      EStatus.HasSideEffects)
    return false;

  return CheckConstantExpression(InitInfo, VD->getLocation(), VD->getType(),
                                 Value);
}

/// isEvaluatable - Call EvaluateAsRValue to see if this expression can be
/// constant folded, but discard the result.
bool Expr::isEvaluatable(const ASTContext &Ctx) const {
  EvalResult Result;
  return EvaluateAsRValue(Result, Ctx) && !Result.HasSideEffects;
}

APSInt Expr::EvaluateKnownConstInt(const ASTContext &Ctx) const {
  EvalResult EvalResult;
  bool Result = EvaluateAsRValue(EvalResult, Ctx);
  (void)Result;
  assert(Result && "Could not evaluate expression");
  assert(EvalResult.Val.isInt() && "Expression did not evaluate to integer");

  return EvalResult.Val.getInt();
}

 bool Expr::EvalResult::isGlobalLValue() const {
   assert(Val.isLValue());
   return IsGlobalLValue(Val.getLValueBase());
 }


/// isIntegerConstantExpr - this recursive routine will test if an expression is
/// an integer constant expression.

/// FIXME: Pass up a reason why! Invalid operation in i-c-e, division by zero,
/// comma, etc
///
/// FIXME: Handle offsetof.  Two things to do:  Handle GCC's __builtin_offsetof
/// to support gcc 4.0+  and handle the idiom GCC recognizes with a null pointer
/// cast+dereference.

// CheckICE - This function does the fundamental ICE checking: the returned
// ICEDiag contains a Val of 0, 1, or 2, and a possibly null SourceLocation.
// Note that to reduce code duplication, this helper does no evaluation
// itself; the caller checks whether the expression is evaluatable, and
// in the rare cases where CheckICE actually cares about the evaluated
// value, it calls into Evalute.
//
// Meanings of Val:
// 0: This expression is an ICE.
// 1: This expression is not an ICE, but if it isn't evaluated, it's
//    a legal subexpression for an ICE. This return value is used to handle
//    the comma operator in C99 mode.
// 2: This expression is not an ICE, and is not a legal subexpression for one.

namespace {

struct ICEDiag {
  unsigned Val;
  SourceLocation Loc;

  public:
  ICEDiag(unsigned v, SourceLocation l) : Val(v), Loc(l) {}
  ICEDiag() : Val(0) {}
};

}

static ICEDiag NoDiag() { return ICEDiag(); }

static ICEDiag CheckEvalInICE(const Expr* E, ASTContext &Ctx) {
  Expr::EvalResult EVResult;
  if (!E->EvaluateAsRValue(EVResult, Ctx) || EVResult.HasSideEffects ||
      !EVResult.Val.isInt()) {
    return ICEDiag(2, E->getLocStart());
  }
  return NoDiag();
}

static ICEDiag CheckICE(const Expr* E, ASTContext &Ctx) {
  assert(!E->isValueDependent() && "Should not see value dependent exprs!");
  if (!E->getType()->isIntegralOrEnumerationType()) {
    return ICEDiag(2, E->getLocStart());
  }

  switch (E->getStmtClass()) {
#define ABSTRACT_STMT(Node)
#define STMT(Node, Base) case Expr::Node##Class:
#define EXPR(Node, Base)
#include "clang/AST/StmtNodes.inc"
  case Expr::PredefinedExprClass:
  case Expr::FloatingLiteralClass:
  case Expr::ImaginaryLiteralClass:
  case Expr::StringLiteralClass:
  case Expr::ArraySubscriptExprClass:
  case Expr::MemberExprClass:
  case Expr::CompoundAssignOperatorClass:
  case Expr::CompoundLiteralExprClass:
  case Expr::ExtVectorElementExprClass:
  case Expr::DesignatedInitExprClass:
  case Expr::ImplicitValueInitExprClass:
  case Expr::ParenListExprClass:
  case Expr::VAArgExprClass:
  case Expr::AddrLabelExprClass:
  case Expr::StmtExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CUDAKernelCallExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::CXXTypeidExprClass:
  case Expr::CXXUuidofExprClass:
  case Expr::CXXNullPtrLiteralExprClass:
  case Expr::UserDefinedLiteralClass:
  case Expr::CXXThisExprClass:
  case Expr::CXXThrowExprClass:
  case Expr::CXXNewExprClass:
  case Expr::CXXDeleteExprClass:
  case Expr::CXXPseudoDestructorExprClass:
  case Expr::UnresolvedLookupExprClass:
  case Expr::DependentScopeDeclRefExprClass:
  case Expr::CXXConstructExprClass:
  case Expr::CXXBindTemporaryExprClass:
  case Expr::ExprWithCleanupsClass:
  case Expr::CXXTemporaryObjectExprClass:
  case Expr::CXXUnresolvedConstructExprClass:
  case Expr::CXXDependentScopeMemberExprClass:
  case Expr::UnresolvedMemberExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::ObjCBoxedExprClass:
  case Expr::ObjCArrayLiteralClass:
  case Expr::ObjCDictionaryLiteralClass:
  case Expr::ObjCEncodeExprClass:
  case Expr::ObjCMessageExprClass:
  case Expr::ObjCSelectorExprClass:
  case Expr::ObjCProtocolExprClass:
  case Expr::ObjCIvarRefExprClass:
  case Expr::ObjCPropertyRefExprClass:
  case Expr::ObjCSubscriptRefExprClass:
  case Expr::ObjCIsaExprClass:
  case Expr::ShuffleVectorExprClass:
  case Expr::BlockExprClass:
  case Expr::NoStmtClass:
  case Expr::OpaqueValueExprClass:
  case Expr::PackExpansionExprClass:
  case Expr::SubstNonTypeTemplateParmPackExprClass:
  case Expr::FunctionParmPackExprClass:
  case Expr::AsTypeExprClass:
  case Expr::ObjCIndirectCopyRestoreExprClass:
  case Expr::MaterializeTemporaryExprClass:
  case Expr::PseudoObjectExprClass:
  case Expr::AtomicExprClass:
  case Expr::InitListExprClass:
  case Expr::LambdaExprClass:
    return ICEDiag(2, E->getLocStart());

  case Expr::SizeOfPackExprClass:
  case Expr::GNUNullExprClass:
    // GCC considers the GNU __null value to be an integral constant expression.
    return NoDiag();

  case Expr::SubstNonTypeTemplateParmExprClass:
    return
      CheckICE(cast<SubstNonTypeTemplateParmExpr>(E)->getReplacement(), Ctx);

  case Expr::ParenExprClass:
    return CheckICE(cast<ParenExpr>(E)->getSubExpr(), Ctx);
  case Expr::GenericSelectionExprClass:
    return CheckICE(cast<GenericSelectionExpr>(E)->getResultExpr(), Ctx);
  case Expr::IntegerLiteralClass:
  case Expr::CharacterLiteralClass:
  case Expr::ObjCBoolLiteralExprClass:
  case Expr::CXXBoolLiteralExprClass:
  case Expr::CXXScalarValueInitExprClass:
  case Expr::UnaryTypeTraitExprClass:
  case Expr::BinaryTypeTraitExprClass:
  case Expr::TypeTraitExprClass:
  case Expr::ArrayTypeTraitExprClass:
  case Expr::ExpressionTraitExprClass:
  case Expr::CXXNoexceptExprClass:
    return NoDiag();
  case Expr::CallExprClass:
  case Expr::CXXOperatorCallExprClass: {
    // C99 6.6/3 allows function calls within unevaluated subexpressions of
    // constant expressions, but they can never be ICEs because an ICE cannot
    // contain an operand of (pointer to) function type.
    const CallExpr *CE = cast<CallExpr>(E);
    if (CE->isBuiltinCall())
      return CheckEvalInICE(E, Ctx);
    return ICEDiag(2, E->getLocStart());
  }
  case Expr::DeclRefExprClass: {
    if (isa<EnumConstantDecl>(cast<DeclRefExpr>(E)->getDecl()))
      return NoDiag();
    const ValueDecl *D = dyn_cast<ValueDecl>(cast<DeclRefExpr>(E)->getDecl());
    if (Ctx.getLangOpts().CPlusPlus &&
        D && IsConstNonVolatile(D->getType())) {
      // Parameter variables are never constants.  Without this check,
      // getAnyInitializer() can find a default argument, which leads
      // to chaos.
      if (isa<ParmVarDecl>(D))
        return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());

      // C++ 7.1.5.1p2
      //   A variable of non-volatile const-qualified integral or enumeration
      //   type initialized by an ICE can be used in ICEs.
      if (const VarDecl *Dcl = dyn_cast<VarDecl>(D)) {
        if (!Dcl->getType()->isIntegralOrEnumerationType())
          return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());

        const VarDecl *VD;
        // Look for a declaration of this variable that has an initializer, and
        // check whether it is an ICE.
        if (Dcl->getAnyInitializer(VD) && VD->checkInitIsICE())
          return NoDiag();
        else
          return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
      }
    }
    return ICEDiag(2, E->getLocStart());
  }
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(E);
    switch (Exp->getOpcode()) {
    case UO_PostInc:
    case UO_PostDec:
    case UO_PreInc:
    case UO_PreDec:
    case UO_AddrOf:
    case UO_Deref:
      // C99 6.6/3 allows increment and decrement within unevaluated
      // subexpressions of constant expressions, but they can never be ICEs
      // because an ICE cannot contain an lvalue operand.
      return ICEDiag(2, E->getLocStart());
    case UO_Extension:
    case UO_LNot:
    case UO_Plus:
    case UO_Minus:
    case UO_Not:
    case UO_Real:
    case UO_Imag:
      return CheckICE(Exp->getSubExpr(), Ctx);
    }
    
    // OffsetOf falls through here.
  }
  case Expr::OffsetOfExprClass: {
      // Note that per C99, offsetof must be an ICE. And AFAIK, using
      // EvaluateAsRValue matches the proposed gcc behavior for cases like
      // "offsetof(struct s{int x[4];}, x[1.0])".  This doesn't affect
      // compliance: we should warn earlier for offsetof expressions with
      // array subscripts that aren't ICEs, and if the array subscripts
      // are ICEs, the value of the offsetof must be an integer constant.
      return CheckEvalInICE(E, Ctx);
  }
  case Expr::UnaryExprOrTypeTraitExprClass: {
    const UnaryExprOrTypeTraitExpr *Exp = cast<UnaryExprOrTypeTraitExpr>(E);
    if ((Exp->getKind() ==  UETT_SizeOf) &&
        Exp->getTypeOfArgument()->isVariableArrayType())
      return ICEDiag(2, E->getLocStart());
    return NoDiag();
  }
  case Expr::BinaryOperatorClass: {
    const BinaryOperator *Exp = cast<BinaryOperator>(E);
    switch (Exp->getOpcode()) {
    case BO_PtrMemD:
    case BO_PtrMemI:
    case BO_Assign:
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      // C99 6.6/3 allows assignments within unevaluated subexpressions of
      // constant expressions, but they can never be ICEs because an ICE cannot
      // contain an lvalue operand.
      return ICEDiag(2, E->getLocStart());

    case BO_Mul:
    case BO_Div:
    case BO_Rem:
    case BO_Add:
    case BO_Sub:
    case BO_Shl:
    case BO_Shr:
    case BO_LT:
    case BO_GT:
    case BO_LE:
    case BO_GE:
    case BO_EQ:
    case BO_NE:
    case BO_And:
    case BO_Xor:
    case BO_Or:
    case BO_Comma: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);
      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (Exp->getOpcode() == BO_Div ||
          Exp->getOpcode() == BO_Rem) {
        // EvaluateAsRValue gives an error for undefined Div/Rem, so make sure
        // we don't evaluate one.
        if (LHSResult.Val == 0 && RHSResult.Val == 0) {
          llvm::APSInt REval = Exp->getRHS()->EvaluateKnownConstInt(Ctx);
          if (REval == 0)
            return ICEDiag(1, E->getLocStart());
          if (REval.isSigned() && REval.isAllOnesValue()) {
            llvm::APSInt LEval = Exp->getLHS()->EvaluateKnownConstInt(Ctx);
            if (LEval.isMinSignedValue())
              return ICEDiag(1, E->getLocStart());
          }
        }
      }
      if (Exp->getOpcode() == BO_Comma) {
        if (Ctx.getLangOpts().C99) {
          // C99 6.6p3 introduces a strange edge case: comma can be in an ICE
          // if it isn't evaluated.
          if (LHSResult.Val == 0 && RHSResult.Val == 0)
            return ICEDiag(1, E->getLocStart());
        } else {
          // In both C89 and C++, commas in ICEs are illegal.
          return ICEDiag(2, E->getLocStart());
        }
      }
      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    case BO_LAnd:
    case BO_LOr: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);
      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (LHSResult.Val == 0 && RHSResult.Val == 1) {
        // Rare case where the RHS has a comma "side-effect"; we need
        // to actually check the condition to see whether the side
        // with the comma is evaluated.
        if ((Exp->getOpcode() == BO_LAnd) !=
            (Exp->getLHS()->EvaluateKnownConstInt(Ctx) == 0))
          return RHSResult;
        return NoDiag();
      }

      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    }
  }
  case Expr::ImplicitCastExprClass:
  case Expr::CStyleCastExprClass:
  case Expr::CXXFunctionalCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass:
  case Expr::ObjCBridgedCastExprClass: {
    const Expr *SubExpr = cast<CastExpr>(E)->getSubExpr();
    if (isa<ExplicitCastExpr>(E)) {
      if (const FloatingLiteral *FL
            = dyn_cast<FloatingLiteral>(SubExpr->IgnoreParenImpCasts())) {
        unsigned DestWidth = Ctx.getIntWidth(E->getType());
        bool DestSigned = E->getType()->isSignedIntegerOrEnumerationType();
        APSInt IgnoredVal(DestWidth, !DestSigned);
        bool Ignored;
        // If the value does not fit in the destination type, the behavior is
        // undefined, so we are not required to treat it as a constant
        // expression.
        if (FL->getValue().convertToInteger(IgnoredVal,
                                            llvm::APFloat::rmTowardZero,
                                            &Ignored) & APFloat::opInvalidOp)
          return ICEDiag(2, E->getLocStart());
        return NoDiag();
      }
    }
    switch (cast<CastExpr>(E)->getCastKind()) {
    case CK_LValueToRValue:
    case CK_AtomicToNonAtomic:
    case CK_NonAtomicToAtomic:
    case CK_NoOp:
    case CK_IntegralToBoolean:
    case CK_IntegralCast:
      return CheckICE(SubExpr, Ctx);
    default:
      return ICEDiag(2, E->getLocStart());
    }
  }
  case Expr::BinaryConditionalOperatorClass: {
    const BinaryConditionalOperator *Exp = cast<BinaryConditionalOperator>(E);
    ICEDiag CommonResult = CheckICE(Exp->getCommon(), Ctx);
    if (CommonResult.Val == 2) return CommonResult;
    ICEDiag FalseResult = CheckICE(Exp->getFalseExpr(), Ctx);
    if (FalseResult.Val == 2) return FalseResult;
    if (CommonResult.Val == 1) return CommonResult;
    if (FalseResult.Val == 1 &&
        Exp->getCommon()->EvaluateKnownConstInt(Ctx) == 0) return NoDiag();
    return FalseResult;
  }
  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(E);
    // If the condition (ignoring parens) is a __builtin_constant_p call,
    // then only the true side is actually considered in an integer constant
    // expression, and it is fully evaluated.  This is an important GNU
    // extension.  See GCC PR38377 for discussion.
    if (const CallExpr *CallCE
        = dyn_cast<CallExpr>(Exp->getCond()->IgnoreParenCasts()))
      if (CallCE->isBuiltinCall() == Builtin::BI__builtin_constant_p)
        return CheckEvalInICE(E, Ctx);
    ICEDiag CondResult = CheckICE(Exp->getCond(), Ctx);
    if (CondResult.Val == 2)
      return CondResult;

    ICEDiag TrueResult = CheckICE(Exp->getTrueExpr(), Ctx);
    ICEDiag FalseResult = CheckICE(Exp->getFalseExpr(), Ctx);

    if (TrueResult.Val == 2)
      return TrueResult;
    if (FalseResult.Val == 2)
      return FalseResult;
    if (CondResult.Val == 1)
      return CondResult;
    if (TrueResult.Val == 0 && FalseResult.Val == 0)
      return NoDiag();
    // Rare case where the diagnostics depend on which side is evaluated
    // Note that if we get here, CondResult is 0, and at least one of
    // TrueResult and FalseResult is non-zero.
    if (Exp->getCond()->EvaluateKnownConstInt(Ctx) == 0) {
      return FalseResult;
    }
    return TrueResult;
  }
  case Expr::CXXDefaultArgExprClass:
    return CheckICE(cast<CXXDefaultArgExpr>(E)->getExpr(), Ctx);
  case Expr::ChooseExprClass: {
    return CheckICE(cast<ChooseExpr>(E)->getChosenSubExpr(Ctx), Ctx);
  }
  }

  llvm_unreachable("Invalid StmtClass!");
}

/// Evaluate an expression as a C++11 integral constant expression.
static bool EvaluateCPlusPlus11IntegralConstantExpr(ASTContext &Ctx,
                                                    const Expr *E,
                                                    llvm::APSInt *Value,
                                                    SourceLocation *Loc) {
  if (!E->getType()->isIntegralOrEnumerationType()) {
    if (Loc) *Loc = E->getExprLoc();
    return false;
  }

  APValue Result;
  if (!E->isCXX11ConstantExpr(Ctx, &Result, Loc))
    return false;

  assert(Result.isInt() && "pointer cast to int is not an ICE");
  if (Value) *Value = Result.getInt();
  return true;
}

bool Expr::isIntegerConstantExpr(ASTContext &Ctx, SourceLocation *Loc) const {
  if (Ctx.getLangOpts().CPlusPlus0x)
    return EvaluateCPlusPlus11IntegralConstantExpr(Ctx, this, 0, Loc);

  ICEDiag d = CheckICE(this, Ctx);
  if (d.Val != 0) {
    if (Loc) *Loc = d.Loc;
    return false;
  }
  return true;
}

bool Expr::isIntegerConstantExpr(llvm::APSInt &Value, ASTContext &Ctx,
                                 SourceLocation *Loc, bool isEvaluated) const {
  if (Ctx.getLangOpts().CPlusPlus0x)
    return EvaluateCPlusPlus11IntegralConstantExpr(Ctx, this, &Value, Loc);

  if (!isIntegerConstantExpr(Ctx, Loc))
    return false;
  if (!EvaluateAsInt(Value, Ctx))
    llvm_unreachable("ICE cannot be evaluated!");
  return true;
}

bool Expr::isCXX98IntegralConstantExpr(ASTContext &Ctx) const {
  return CheckICE(this, Ctx).Val == 0;
}

bool Expr::isCXX11ConstantExpr(ASTContext &Ctx, APValue *Result,
                               SourceLocation *Loc) const {
  // We support this checking in C++98 mode in order to diagnose compatibility
  // issues.
  assert(Ctx.getLangOpts().CPlusPlus);

  // Build evaluation settings.
  Expr::EvalStatus Status;
  llvm::SmallVector<PartialDiagnosticAt, 8> Diags;
  Status.Diag = &Diags;
  EvalInfo Info(Ctx, Status);

  APValue Scratch;
  bool IsConstExpr = ::EvaluateAsRValue(Info, this, Result ? *Result : Scratch);

  if (!Diags.empty()) {
    IsConstExpr = false;
    if (Loc) *Loc = Diags[0].first;
  } else if (!IsConstExpr) {
    // FIXME: This shouldn't happen.
    if (Loc) *Loc = getExprLoc();
  }

  return IsConstExpr;
}

bool Expr::isPotentialConstantExpr(const FunctionDecl *FD,
                                   llvm::SmallVectorImpl<
                                     PartialDiagnosticAt> &Diags) {
  // FIXME: It would be useful to check constexpr function templates, but at the
  // moment the constant expression evaluator cannot cope with the non-rigorous
  // ASTs which we build for dependent expressions.
  if (FD->isDependentContext())
    return true;

  Expr::EvalStatus Status;
  Status.Diag = &Diags;

  EvalInfo Info(FD->getASTContext(), Status);
  Info.CheckingPotentialConstantExpression = true;

  const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
  const CXXRecordDecl *RD = MD ? MD->getParent()->getCanonicalDecl() : 0;

  // FIXME: Fabricate an arbitrary expression on the stack and pretend that it
  // is a temporary being used as the 'this' pointer.
  LValue This;
  ImplicitValueInitExpr VIE(RD ? Info.Ctx.getRecordType(RD) : Info.Ctx.IntTy);
  This.set(&VIE, Info.CurrentCall->Index);

  ArrayRef<const Expr*> Args;

  SourceLocation Loc = FD->getLocation();

  APValue Scratch;
  if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(FD))
    HandleConstructorCall(Loc, This, Args, CD, Info, Scratch);
  else
    HandleFunctionCall(Loc, FD, (MD && MD->isInstance()) ? &This : 0,
                       Args, FD->getBody(), Info, Scratch);

  return Diags.empty();
}
