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
#include "llvm/ADT/Optional.h"

namespace clang {
class QualType;

namespace interp {

template <class Emitter> class LocalScope;
template <class Emitter> class RecordScope;
template <class Emitter> class VariableScope;
template <class Emitter> class DeclScope;
template <class Emitter> class OptionScope;
template <class Emitter> class ArrayIndexScope;

/// Compilation context for expressions.
template <class Emitter>
class ByteCodeExprGen : public ConstStmtVisitor<ByteCodeExprGen<Emitter>, bool>,
                        public Emitter {
protected:
  // Emitters for opcodes of various arities.
  using NullaryFn = bool (ByteCodeExprGen::*)(const SourceInfo &);
  using UnaryFn = bool (ByteCodeExprGen::*)(PrimType, const SourceInfo &);
  using BinaryFn = bool (ByteCodeExprGen::*)(PrimType, PrimType,
                                             const SourceInfo &);

  // Aliases for types defined in the emitter.
  using LabelTy = typename Emitter::LabelTy;
  using AddrTy = typename Emitter::AddrTy;

  // Reference to a function generating the pointer of an initialized object.s
  using InitFnRef = std::function<bool()>;

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
  bool VisitParenExpr(const ParenExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *E);
  bool VisitCallExpr(const CallExpr *E);
  bool VisitCXXMemberCallExpr(const CXXMemberCallExpr *E);
  bool VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *E);
  bool VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E);
  bool VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *E);
  bool VisitCXXThisExpr(const CXXThisExpr *E);
  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitDeclRefExpr(const DeclRefExpr *E);
  bool VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E);
  bool VisitSubstNonTypeTemplateParmExpr(const SubstNonTypeTemplateParmExpr *E);
  bool VisitArraySubscriptExpr(const ArraySubscriptExpr *E);
  bool VisitInitListExpr(const InitListExpr *E);
  bool VisitConstantExpr(const ConstantExpr *E);
  bool VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E);
  bool VisitMemberExpr(const MemberExpr *E);
  bool VisitArrayInitIndexExpr(const ArrayInitIndexExpr *E);
  bool VisitOpaqueValueExpr(const OpaqueValueExpr *E);
  bool VisitAbstractConditionalOperator(const AbstractConditionalOperator *E);

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

  /// Returns the size int bits of an integer.
  unsigned getIntWidth(QualType Ty) {
    auto &ASTContext = Ctx.getASTContext();
    return ASTContext.getIntWidth(Ty);
  }

  /// Returns the value of CHAR_BIT.
  unsigned getCharBit() const {
    auto &ASTContext = Ctx.getASTContext();
    return ASTContext.getTargetInfo().getCharWidth();
  }

  /// Classifies a type.
  llvm::Optional<PrimType> classify(const Expr *E) const {
    return E->isGLValue() ? PT_Ptr : classify(E->getType());
  }
  llvm::Optional<PrimType> classify(QualType Ty) const {
    return Ctx.classify(Ty);
  }

  /// Checks if a pointer needs adjustment.
  bool needsAdjust(QualType Ty) const {
    return true;
  }

  /// Classifies a known primitive type
  PrimType classifyPrim(QualType Ty) const {
    if (auto T = classify(Ty)) {
      return *T;
    }
    llvm_unreachable("not a primitive type");
  }

  /// Evaluates an expression for side effects and discards the result.
  bool discard(const Expr *E);
  /// Evaluates an expression and places result on stack.
  bool visit(const Expr *E);
  /// Compiles an initializer.
  bool visitInitializer(const Expr *E);
  /// Compiles an array initializer.
  bool visitArrayInitializer(const Expr *Initializer);
  /// Compiles a record initializer.
  bool visitRecordInitializer(const Expr *Initializer);

  /// Visits an expression and converts it to a boolean.
  bool visitBool(const Expr *E);

  /// Visits an initializer for a local.
  bool visitLocalInitializer(const Expr *Init, unsigned I) {
    if (!this->emitGetPtrLocal(I, Init))
      return false;

    if (!visitInitializer(Init))
      return false;

    return this->emitPopPtr(Init);
  }

  /// Visits an initializer for a global.
  bool visitGlobalInitializer(const Expr *Init, unsigned I) {
    if (!this->emitGetPtrGlobal(I, Init))
      return false;

    if (!visitInitializer(Init))
      return false;

    return this->emitPopPtr(Init);
  }

  /// Visits a delegated initializer.
  bool visitThisInitializer(const Expr *I) {
    if (!this->emitThis(I))
      return false;

    if (!visitInitializer(I))
      return false;

    return this->emitPopPtr(I);
  }

  /// Creates a local primitive value.
  unsigned allocateLocalPrimitive(DeclTy &&Decl, PrimType Ty, bool IsMutable,
                                  bool IsExtended = false);

  /// Allocates a space storing a local given its type.
  llvm::Optional<unsigned> allocateLocal(DeclTy &&Decl,
                                         bool IsExtended = false);

private:
  friend class VariableScope<Emitter>;
  friend class LocalScope<Emitter>;
  friend class RecordScope<Emitter>;
  friend class DeclScope<Emitter>;
  friend class OptionScope<Emitter>;
  friend class ArrayIndexScope<Emitter>;

  /// Emits a zero initializer.
  bool visitZeroInitializer(PrimType T, const Expr *E);

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

  /// Emits an APInt constant.
  bool emitConst(PrimType T, const llvm::APInt &Value, const Expr *E);

  /// Emits an integer constant.
  template <typename T> bool emitConst(const Expr *E, T Value) {
    QualType Ty = E->getType();
    APInt WrappedValue(getIntWidth(Ty), Value, std::is_signed<T>::value);
    return emitConst(*Ctx.classify(Ty), WrappedValue, E);
  }

  /// Returns the index of a global.
  llvm::Optional<unsigned> getGlobalIdx(const VarDecl *VD);

  /// Emits the initialized pointer.
  bool emitInitFn() {
    assert(InitFn && "missing initializer");
    return (*InitFn)();
  }

protected:
  /// Variable to storage mapping.
  llvm::DenseMap<const ValueDecl *, Scope::Local> Locals;

  /// OpaqueValueExpr to location mapping.
  llvm::DenseMap<const OpaqueValueExpr *, unsigned> OpaqueExprs;

  /// Current scope.
  VariableScope<Emitter> *VarScope = nullptr;

  /// Current argument index. Needed to emit ArrayInitIndexExpr.
  llvm::Optional<uint64_t> ArrayIndex;

  /// Flag indicating if return value is to be discarded.
  bool DiscardResult = false;

  /// Expression being initialized.
  llvm::Optional<InitFnRef> InitFn = {};
};

extern template class ByteCodeExprGen<ByteCodeEmitter>;
extern template class ByteCodeExprGen<EvalEmitter>;

/// Scope chain managing the variable lifetimes.
template <class Emitter> class VariableScope {
public:
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

  VariableScope *getParent() { return Parent; }

protected:
  VariableScope(ByteCodeExprGen<Emitter> *Ctx)
      : Ctx(Ctx), Parent(Ctx->VarScope) {
    Ctx->VarScope = this;
  }

  /// ByteCodeExprGen instance.
  ByteCodeExprGen<Emitter> *Ctx;
  /// Link to the parent scope.
  VariableScope *Parent;
};

/// Scope for local variables.
///
/// When the scope is destroyed, instructions are emitted to tear down
/// all variables declared in this scope.
template <class Emitter> class LocalScope : public VariableScope<Emitter> {
public:
  LocalScope(ByteCodeExprGen<Emitter> *Ctx) : VariableScope<Emitter>(Ctx) {}

  ~LocalScope() override { this->emitDestruction(); }

  void addLocal(const Scope::Local &Local) override {
    if (!Idx) {
      Idx = this->Ctx->Descriptors.size();
      this->Ctx->Descriptors.emplace_back();
    }

    this->Ctx->Descriptors[*Idx].emplace_back(Local);
  }

  void emitDestruction() override {
    if (!Idx)
      return;
    this->Ctx->emitDestroy(*Idx, SourceInfo{});
  }

protected:
  /// Index of the scope in the chain.
  Optional<unsigned> Idx;
};

/// Scope for storage declared in a compound statement.
template <class Emitter> class BlockScope final : public LocalScope<Emitter> {
public:
  BlockScope(ByteCodeExprGen<Emitter> *Ctx) : LocalScope<Emitter>(Ctx) {}

  void addExtended(const Scope::Local &Local) override {
    llvm_unreachable("Cannot create temporaries in full scopes");
  }
};

/// Expression scope which tracks potentially lifetime extended
/// temporaries which are hoisted to the parent scope on exit.
template <class Emitter> class ExprScope final : public LocalScope<Emitter> {
public:
  ExprScope(ByteCodeExprGen<Emitter> *Ctx) : LocalScope<Emitter>(Ctx) {}

  void addExtended(const Scope::Local &Local) override {
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
  Optional<uint64_t> OldArrayIndex;
};

} // namespace interp
} // namespace clang

#endif
