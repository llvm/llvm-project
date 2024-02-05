//===--- ByteCodeExprGen.h - Code generator for expressions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the constexpr bytecode compiler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_BYTECODEEXPRGEN_H
#define LLVM_CLANG_AST_INTERP_BYTECODEEXPRGEN_H

#include "ByteCodeEmitter.h"
#include "EvalEmitter.h"
#include "Pointer.h"
#include "PrimType.h"
#include "Record.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/TargetInfo.h"

namespace clang {
class QualType;

namespace interp {

template <class Emitter> class LocalScope;
template <class Emitter> class DestructorScope;
template <class Emitter> class VariableScope;
template <class Emitter> class DeclScope;
template <class Emitter> class OptionScope;
template <class Emitter> class ArrayIndexScope;
template <class Emitter> class SourceLocScope;

/// Compilation context for expressions.
template <class Emitter>
class ByteCodeExprGen : public ConstStmtVisitor<ByteCodeExprGen<Emitter>, bool>,
                        public Emitter {
protected:
  // Aliases for types defined in the emitter.
  using LabelTy = typename Emitter::LabelTy;
  using AddrTy = typename Emitter::AddrTy;

  /// Current compilation context.
  Context &Ctx;
  /// Program to link to.
  Program &P;

public:
  /// Initializes the compiler and the backend emitter.
  template <typename... Tys>
  ByteCodeExprGen(Context &Ctx, Program &P, Tys &&... Args)
      : Emitter(Ctx, P, Args...), Ctx(Ctx), P(P) {}

  // Expression visitors - result returned on interp stack.
  bool VisitCastExpr(const CastExpr *E);
  bool VisitIntegerLiteral(const IntegerLiteral *E);
  bool VisitFloatingLiteral(const FloatingLiteral *E);
  bool VisitImaginaryLiteral(const ImaginaryLiteral *E);
  bool VisitParenExpr(const ParenExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitLogicalBinOp(const BinaryOperator *E);
  bool VisitPointerArithBinOp(const BinaryOperator *E);
  bool VisitComplexBinOp(const BinaryOperator *E);
  bool VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *E);
  bool VisitCallExpr(const CallExpr *E);
  bool VisitBuiltinCallExpr(const CallExpr *E);
  bool VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *E);
  bool VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E);
  bool VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *E);
  bool VisitGNUNullExpr(const GNUNullExpr *E);
  bool VisitCXXThisExpr(const CXXThisExpr *E);
  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitDeclRefExpr(const DeclRefExpr *E);
  bool VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E);
  bool VisitSubstNonTypeTemplateParmExpr(const SubstNonTypeTemplateParmExpr *E);
  bool VisitArraySubscriptExpr(const ArraySubscriptExpr *E);
  bool VisitInitListExpr(const InitListExpr *E);
  bool VisitCXXParenListInitExpr(const CXXParenListInitExpr *E);
  bool VisitConstantExpr(const ConstantExpr *E);
  bool VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E);
  bool VisitMemberExpr(const MemberExpr *E);
  bool VisitArrayInitIndexExpr(const ArrayInitIndexExpr *E);
  bool VisitArrayInitLoopExpr(const ArrayInitLoopExpr *E);
  bool VisitOpaqueValueExpr(const OpaqueValueExpr *E);
  bool VisitAbstractConditionalOperator(const AbstractConditionalOperator *E);
  bool VisitStringLiteral(const StringLiteral *E);
  bool VisitCharacterLiteral(const CharacterLiteral *E);
  bool VisitCompoundAssignOperator(const CompoundAssignOperator *E);
  bool VisitFloatCompoundAssignOperator(const CompoundAssignOperator *E);
  bool VisitPointerCompoundAssignOperator(const CompoundAssignOperator *E);
  bool VisitExprWithCleanups(const ExprWithCleanups *E);
  bool VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E);
  bool VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *E);
  bool VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
  bool VisitTypeTraitExpr(const TypeTraitExpr *E);
  bool VisitLambdaExpr(const LambdaExpr *E);
  bool VisitPredefinedExpr(const PredefinedExpr *E);
  bool VisitCXXThrowExpr(const CXXThrowExpr *E);
  bool VisitCXXReinterpretCastExpr(const CXXReinterpretCastExpr *E);
  bool VisitCXXNoexceptExpr(const CXXNoexceptExpr *E);
  bool VisitCXXConstructExpr(const CXXConstructExpr *E);
  bool VisitSourceLocExpr(const SourceLocExpr *E);
  bool VisitOffsetOfExpr(const OffsetOfExpr *E);
  bool VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E);
  bool VisitSizeOfPackExpr(const SizeOfPackExpr *E);
  bool VisitGenericSelectionExpr(const GenericSelectionExpr *E);
  bool VisitChooseExpr(const ChooseExpr *E);
  bool VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *E);

protected:
  bool visitExpr(const Expr *E) override;
  bool visitDecl(const VarDecl *VD) override;

protected:
  /// Emits scope cleanup instructions.
  void emitCleanup();

  /// Returns a record type from a record or pointer type.
  const RecordType *getRecordTy(QualType Ty);

  /// Returns a record from a record or pointer type.
  Record *getRecord(QualType Ty);
  Record *getRecord(const RecordDecl *RD);

  // Returns a function for the given FunctionDecl.
  // If the function does not exist yet, it is compiled.
  const Function *getFunction(const FunctionDecl *FD);

  std::optional<PrimType> classify(const Expr *E) const {
    return Ctx.classify(E);
  }
  std::optional<PrimType> classify(QualType Ty) const {
    return Ctx.classify(Ty);
  }

  /// Classifies a known primitive type
  PrimType classifyPrim(QualType Ty) const {
    if (auto T = classify(Ty)) {
      return *T;
    }
    llvm_unreachable("not a primitive type");
  }
  /// Evaluates an expression and places the result on the stack. If the
  /// expression is of composite type, a local variable will be created
  /// and a pointer to said variable will be placed on the stack.
  bool visit(const Expr *E);
  /// Compiles an initializer. This is like visit() but it will never
  /// create a variable and instead rely on a variable already having
  /// been created. visitInitializer() then relies on a pointer to this
  /// variable being on top of the stack.
  bool visitInitializer(const Expr *E);
  /// Evaluates an expression for side effects and discards the result.
  bool discard(const Expr *E);
  /// Just pass evaluation on to \p E. This leaves all the parsing flags
  /// intact.
  bool delegate(const Expr *E);

  /// Creates and initializes a variable from the given decl.
  bool visitVarDecl(const VarDecl *VD);
  /// Visit an APValue.
  bool visitAPValue(const APValue &Val, PrimType ValType, const Expr *E);

  /// Visits an expression and converts it to a boolean.
  bool visitBool(const Expr *E);

  /// Visits an initializer for a local.
  bool visitLocalInitializer(const Expr *Init, unsigned I) {
    if (!this->emitGetPtrLocal(I, Init))
      return false;

    if (!visitInitializer(Init))
      return false;

    if (!this->emitInitPtr(Init))
      return false;

    return this->emitPopPtr(Init);
  }

  /// Visits an initializer for a global.
  bool visitGlobalInitializer(const Expr *Init, unsigned I) {
    if (!this->emitGetPtrGlobal(I, Init))
      return false;

    if (!visitInitializer(Init))
      return false;

    if (!this->emitInitPtr(Init))
      return false;

    return this->emitPopPtr(Init);
  }

  /// Visits a delegated initializer.
  bool visitThisInitializer(const Expr *I) {
    if (!this->emitThis(I))
      return false;

    if (!visitInitializer(I))
      return false;

    return this->emitInitPtrPop(I);
  }

  bool visitInitList(ArrayRef<const Expr *> Inits, const Expr *E);
  bool visitArrayElemInit(unsigned ElemIndex, const Expr *Init);

  /// Creates a local primitive value.
  unsigned allocateLocalPrimitive(DeclTy &&Decl, PrimType Ty, bool IsConst,
                                  bool IsExtended = false);

  /// Allocates a space storing a local given its type.
  std::optional<unsigned> allocateLocal(DeclTy &&Decl, bool IsExtended = false);

private:
  friend class VariableScope<Emitter>;
  friend class LocalScope<Emitter>;
  friend class DestructorScope<Emitter>;
  friend class DeclScope<Emitter>;
  friend class OptionScope<Emitter>;
  friend class ArrayIndexScope<Emitter>;
  friend class SourceLocScope<Emitter>;

  /// Emits a zero initializer.
  bool visitZeroInitializer(PrimType T, QualType QT, const Expr *E);
  bool visitZeroRecordInitializer(const Record *R, const Expr *E);

  enum class DerefKind {
    /// Value is read and pushed to stack.
    Read,
    /// Direct method generates a value which is written. Returns pointer.
    Write,
    /// Direct method receives the value, pushes mutated value. Returns pointer.
    ReadWrite,
  };

  /// Method to directly load a value. If the value can be fetched directly,
  /// the direct handler is called. Otherwise, a pointer is left on the stack
  /// and the indirect handler is expected to operate on that.
  bool dereference(const Expr *LV, DerefKind AK,
                   llvm::function_ref<bool(PrimType)> Direct,
                   llvm::function_ref<bool(PrimType)> Indirect);
  bool dereferenceParam(const Expr *LV, PrimType T, const ParmVarDecl *PD,
                        DerefKind AK,
                        llvm::function_ref<bool(PrimType)> Direct,
                        llvm::function_ref<bool(PrimType)> Indirect);
  bool dereferenceVar(const Expr *LV, PrimType T, const VarDecl *PD,
                      DerefKind AK, llvm::function_ref<bool(PrimType)> Direct,
                      llvm::function_ref<bool(PrimType)> Indirect);

  /// Emits an APSInt constant.
  bool emitConst(const llvm::APSInt &Value, PrimType Ty, const Expr *E);
  bool emitConst(const llvm::APSInt &Value, const Expr *E);
  bool emitConst(const llvm::APInt &Value, const Expr *E) {
    return emitConst(static_cast<llvm::APSInt>(Value), E);
  }

  /// Emits an integer constant.
  template <typename T> bool emitConst(T Value, PrimType Ty, const Expr *E);
  template <typename T> bool emitConst(T Value, const Expr *E);

  llvm::RoundingMode getRoundingMode(const Expr *E) const {
    FPOptions FPO = E->getFPFeaturesInEffect(Ctx.getLangOpts());

    if (FPO.getRoundingMode() == llvm::RoundingMode::Dynamic)
      return llvm::RoundingMode::NearestTiesToEven;

    return FPO.getRoundingMode();
  }

  bool emitPrimCast(PrimType FromT, PrimType ToT, QualType ToQT, const Expr *E);
  PrimType classifyComplexElementType(QualType T) const {
    assert(T->isAnyComplexType());

    QualType ElemType = T->getAs<ComplexType>()->getElementType();

    return *this->classify(ElemType);
  }

  bool emitComplexReal(const Expr *SubExpr);

  bool emitRecordDestruction(const Descriptor *Desc);
  unsigned collectBaseOffset(const RecordType *BaseType,
                             const RecordType *DerivedType);

protected:
  /// Variable to storage mapping.
  llvm::DenseMap<const ValueDecl *, Scope::Local> Locals;

  /// OpaqueValueExpr to location mapping.
  llvm::DenseMap<const OpaqueValueExpr *, unsigned> OpaqueExprs;

  /// Current scope.
  VariableScope<Emitter> *VarScope = nullptr;

  /// Current argument index. Needed to emit ArrayInitIndexExpr.
  std::optional<uint64_t> ArrayIndex;

  /// DefaultInit- or DefaultArgExpr, needed for SourceLocExpr.
  const Expr *SourceLocDefaultExpr = nullptr;

  /// Flag indicating if return value is to be discarded.
  bool DiscardResult = false;

  /// Flag inidicating if we're initializing an already created
  /// variable. This is set in visitInitializer().
  bool Initializing = false;

  /// Flag indicating if we're initializing a global variable.
  bool GlobalDecl = false;
};

extern template class ByteCodeExprGen<ByteCodeEmitter>;
extern template class ByteCodeExprGen<EvalEmitter>;

/// Scope chain managing the variable lifetimes.
template <class Emitter> class VariableScope {
public:
  VariableScope(ByteCodeExprGen<Emitter> *Ctx)
      : Ctx(Ctx), Parent(Ctx->VarScope) {
    Ctx->VarScope = this;
  }

  virtual ~VariableScope() { Ctx->VarScope = this->Parent; }

  void add(const Scope::Local &Local, bool IsExtended) {
    if (IsExtended)
      this->addExtended(Local);
    else
      this->addLocal(Local);
  }

  virtual void addLocal(const Scope::Local &Local) {
    if (this->Parent)
      this->Parent->addLocal(Local);
  }

  virtual void addExtended(const Scope::Local &Local) {
    if (this->Parent)
      this->Parent->addExtended(Local);
  }

  virtual void emitDestruction() {}
  virtual void emitDestructors() {}
  VariableScope *getParent() const { return Parent; }

protected:
  /// ByteCodeExprGen instance.
  ByteCodeExprGen<Emitter> *Ctx;
  /// Link to the parent scope.
  VariableScope *Parent;
};

/// Generic scope for local variables.
template <class Emitter> class LocalScope : public VariableScope<Emitter> {
public:
  LocalScope(ByteCodeExprGen<Emitter> *Ctx) : VariableScope<Emitter>(Ctx) {}

  /// Emit a Destroy op for this scope.
  ~LocalScope() override {
    if (!Idx)
      return;
    this->Ctx->emitDestroy(*Idx, SourceInfo{});
    removeStoredOpaqueValues();
  }

  /// Overriden to support explicit destruction.
  void emitDestruction() override {
    if (!Idx)
      return;
    this->emitDestructors();
    this->Ctx->emitDestroy(*Idx, SourceInfo{});
    removeStoredOpaqueValues();
    this->Idx = std::nullopt;
  }

  void addLocal(const Scope::Local &Local) override {
    if (!Idx) {
      Idx = this->Ctx->Descriptors.size();
      this->Ctx->Descriptors.emplace_back();
    }

    this->Ctx->Descriptors[*Idx].emplace_back(Local);
  }

  void emitDestructors() override {
    if (!Idx)
      return;
    // Emit destructor calls for local variables of record
    // type with a destructor.
    for (Scope::Local &Local : this->Ctx->Descriptors[*Idx]) {
      if (!Local.Desc->isPrimitive() && !Local.Desc->isPrimitiveArray()) {
        this->Ctx->emitGetPtrLocal(Local.Offset, SourceInfo{});
        this->Ctx->emitRecordDestruction(Local.Desc);
        removeIfStoredOpaqueValue(Local);
      }
    }
  }

  void removeStoredOpaqueValues() {
    if (!Idx)
      return;

    for (const Scope::Local &Local : this->Ctx->Descriptors[*Idx]) {
      removeIfStoredOpaqueValue(Local);
    }
  }

  void removeIfStoredOpaqueValue(const Scope::Local &Local) {
    if (const auto *OVE =
            llvm::dyn_cast_if_present<OpaqueValueExpr>(Local.Desc->asExpr())) {
      if (auto It = this->Ctx->OpaqueExprs.find(OVE);
          It != this->Ctx->OpaqueExprs.end())
        this->Ctx->OpaqueExprs.erase(It);
    };
  }

  /// Index of the scope in the chain.
  std::optional<unsigned> Idx;
};

/// Emits the destructors of the variables of \param OtherScope
/// when this scope is destroyed. Does not create a Scope in the bytecode at
/// all, this is just a RAII object to emit destructors.
template <class Emitter> class DestructorScope final {
public:
  DestructorScope(LocalScope<Emitter> &OtherScope) : OtherScope(OtherScope) {}

  ~DestructorScope() { OtherScope.emitDestructors(); }

private:
  LocalScope<Emitter> &OtherScope;
};

/// Like a regular LocalScope, except that the destructors of all local
/// variables are automatically emitted when the AutoScope is destroyed.
template <class Emitter> class AutoScope : public LocalScope<Emitter> {
public:
  AutoScope(ByteCodeExprGen<Emitter> *Ctx)
      : LocalScope<Emitter>(Ctx), DS(*this) {}

private:
  DestructorScope<Emitter> DS;
};

/// Scope for storage declared in a compound statement.
template <class Emitter> class BlockScope final : public AutoScope<Emitter> {
public:
  BlockScope(ByteCodeExprGen<Emitter> *Ctx) : AutoScope<Emitter>(Ctx) {}

  void addExtended(const Scope::Local &Local) override {
    // If we to this point, just add the variable as a normal local
    // variable. It will be destroyed at the end of the block just
    // like all others.
    this->addLocal(Local);
  }
};

/// Expression scope which tracks potentially lifetime extended
/// temporaries which are hoisted to the parent scope on exit.
template <class Emitter> class ExprScope final : public AutoScope<Emitter> {
public:
  ExprScope(ByteCodeExprGen<Emitter> *Ctx) : AutoScope<Emitter>(Ctx) {}

  void addExtended(const Scope::Local &Local) override {
    if (this->Parent)
      this->Parent->addLocal(Local);
  }
};

template <class Emitter> class ArrayIndexScope final {
public:
  ArrayIndexScope(ByteCodeExprGen<Emitter> *Ctx, uint64_t Index) : Ctx(Ctx) {
    OldArrayIndex = Ctx->ArrayIndex;
    Ctx->ArrayIndex = Index;
  }

  ~ArrayIndexScope() { Ctx->ArrayIndex = OldArrayIndex; }

private:
  ByteCodeExprGen<Emitter> *Ctx;
  std::optional<uint64_t> OldArrayIndex;
};

template <class Emitter> class SourceLocScope final {
public:
  SourceLocScope(ByteCodeExprGen<Emitter> *Ctx, const Expr *DefaultExpr)
      : Ctx(Ctx) {
    assert(DefaultExpr);
    // We only switch if the current SourceLocDefaultExpr is null.
    if (!Ctx->SourceLocDefaultExpr) {
      Enabled = true;
      Ctx->SourceLocDefaultExpr = DefaultExpr;
    }
  }

  ~SourceLocScope() {
    if (Enabled)
      Ctx->SourceLocDefaultExpr = nullptr;
  }

private:
  ByteCodeExprGen<Emitter> *Ctx;
  bool Enabled = false;
};

} // namespace interp
} // namespace clang

#endif
