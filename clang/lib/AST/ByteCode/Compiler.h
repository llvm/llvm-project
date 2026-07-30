//===--- Compiler.h - Code generator for expressions -----*- C++ -*-===//
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

namespace clang {
class QualType;

namespace interp {

template <class Emitter> class LocalScope;
template <class Emitter> class DestructorScope;
template <class Emitter> class VariableScope;
template <class Emitter> class DeclScope;
template <class Emitter> class InitLinkScope;
template <class Emitter> class InitStackScope;
template <class Emitter> class OptionScope;
template <class Emitter> class ArrayIndexScope;
template <class Emitter> class SourceLocScope;
template <class Emitter> class LoopScope;
template <class Emitter> class LabelScope;
template <class Emitter> class SwitchScope;
template <class Emitter> class StmtExprScope;
template <class Emitter> class LocOverrideScope;

template <class Emitter> class Compiler;
struct InitLink {
public:
  enum {
    K_This = 0,
    K_Field = 1,
    K_Temp = 2,
    K_Decl = 3,
    K_Elem = 5,
    K_RVO = 6,
    K_InitList = 7,
    K_DIE = 8,
  };

  static InitLink This() { return InitLink{K_This}; }
  static InitLink InitList() { return InitLink{K_InitList}; }
  static InitLink RVO() { return InitLink{K_RVO}; }
  static InitLink DIE() { return InitLink{K_DIE}; }
  static InitLink Field(unsigned Offset) {
    InitLink IL{K_Field};
    IL.Offset = Offset;
    return IL;
  }
  static InitLink Temp(unsigned Offset) {
    InitLink IL{K_Temp};
    IL.Offset = Offset;
    return IL;
  }
  static InitLink Decl(const ValueDecl *D) {
    InitLink IL{K_Decl};
    IL.D = D;
    return IL;
  }
  static InitLink Elem(unsigned Index) {
    InitLink IL{K_Elem};
    IL.Offset = Index;
    return IL;
  }

  InitLink(uint8_t Kind) : Kind(Kind) {}
  template <class Emitter>
  bool emit(Compiler<Emitter> *Ctx, const Expr *E) const;

  uint32_t Kind;
  union {
    unsigned Offset;
    const ValueDecl *D;
  };
};

/// State encapsulating if a the variable creation has been successful,
/// unsuccessful, or no variable has been created at all.
struct VarCreationState {
  std::optional<bool> S = std::nullopt;
  VarCreationState() = default;
  VarCreationState(bool b) : S(b) {}
  static VarCreationState NotCreated() { return VarCreationState(); }

  operator bool() const { return S && *S; }
  bool notCreated() const { return !S; }
};

enum class ScopeKind { Block, FullExpression, Call };

/// Compilation context for expressions.
template <class Emitter>
class Compiler : public ConstStmtVisitor<Compiler<Emitter>, bool>,
                 public Emitter {
protected:
  // Aliases for types defined in the emitter.
  using LabelTy = typename Emitter::LabelTy;
  using AddrTy = typename Emitter::AddrTy;
  using OptLabelTy = UnsignedOrNone;
  using CaseMap = llvm::DenseMap<const SwitchCase *, LabelTy>;

  struct LabelInfo {
    const Stmt *Name;
    const VariableScope<Emitter> *BreakOrContinueScope;
    OptLabelTy BreakLabel;
    OptLabelTy ContinueLabel;
    OptLabelTy DefaultLabel;
    LabelInfo(const Stmt *Name, OptLabelTy BreakLabel, OptLabelTy ContinueLabel,
              OptLabelTy DefaultLabel,
              const VariableScope<Emitter> *BreakOrContinueScope)
        : Name(Name), BreakOrContinueScope(BreakOrContinueScope),
          BreakLabel(BreakLabel), ContinueLabel(ContinueLabel),
          DefaultLabel(DefaultLabel) {}
  };

  /// Current compilation context.
  Context &Ctx;
  /// Program to link to.
  Program &P;

public:
  /// Initializes the compiler and the backend emitter.
  template <typename... Tys>
  Compiler(Context &Ctx, Program &P, Tys &&...Args)
      : Emitter(Ctx, P, Args...), Ctx(Ctx), P(P) {}

  // Expressions.
  bool VisitCastExpr(const CastExpr *E);
  bool VisitBuiltinBitCastExpr(const BuiltinBitCastExpr *E);
  bool VisitIntegerLiteral(const IntegerLiteral *E);
  bool VisitFloatingLiteral(const FloatingLiteral *E);
  bool VisitImaginaryLiteral(const ImaginaryLiteral *E);
  bool VisitFixedPointLiteral(const FixedPointLiteral *E);
  bool VisitParenExpr(const ParenExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitLogicalBinOp(const BinaryOperator *E);
  bool VisitPointerArithBinOp(const BinaryOperator *E);
  bool VisitComplexBinOp(const BinaryOperator *E);
  bool VisitVectorBinOp(const BinaryOperator *E);
  bool VisitFixedPointBinOp(const BinaryOperator *E);
  bool VisitFixedPointUnaryOperator(const UnaryOperator *E);
  bool VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *E);
  bool VisitCallExpr(const CallExpr *E);
  bool VisitBuiltinCallExpr(const CallExpr *E, unsigned BuiltinID);
  bool VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *E);
  bool VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E);
  bool VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *E);
  bool VisitGNUNullExpr(const GNUNullExpr *E);
  bool VisitCXXThisExpr(const CXXThisExpr *E);
  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitVectorUnaryOperator(const UnaryOperator *E);
  bool VisitComplexUnaryOperator(const UnaryOperator *E);
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
  bool VisitObjCStringLiteral(const ObjCStringLiteral *E);
  bool VisitObjCEncodeExpr(const ObjCEncodeExpr *E);
  bool VisitSYCLUniqueStableNameExpr(const SYCLUniqueStableNameExpr *E);
  bool VisitCharacterLiteral(const CharacterLiteral *E);
  bool VisitCompoundAssignOperator(const CompoundAssignOperator *E);
  bool VisitFloatCompoundAssignOperator(const CompoundAssignOperator *E);
  bool VisitPointerCompoundAssignOperator(const CompoundAssignOperator *E);
  bool VisitExprWithCleanups(const ExprWithCleanups *E);
  bool VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E);
  bool VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *E);
  bool VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
  bool VisitTypeTraitExpr(const TypeTraitExpr *E);
  bool VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *E);
  bool VisitLambdaExpr(const LambdaExpr *E);
  bool VisitPredefinedExpr(const PredefinedExpr *E);
  bool VisitCXXThrowExpr(const CXXThrowExpr *E);
  bool VisitCXXReinterpretCastExpr(const CXXReinterpretCastExpr *E);
  bool VisitCXXDynamicCastExpr(const CXXDynamicCastExpr *E);
  bool VisitCXXNoexceptExpr(const CXXNoexceptExpr *E);
  bool VisitCXXConstructExpr(const CXXConstructExpr *E);
  bool VisitSourceLocExpr(const SourceLocExpr *E);
  bool VisitOffsetOfExpr(const OffsetOfExpr *E);
  bool VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E);
  bool VisitSizeOfPackExpr(const SizeOfPackExpr *E);
  bool VisitGenericSelectionExpr(const GenericSelectionExpr *E);
  bool VisitChooseExpr(const ChooseExpr *E);
  bool VisitEmbedExpr(const EmbedExpr *E);
  bool VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *E);
  bool VisitCXXInheritedCtorInitExpr(const CXXInheritedCtorInitExpr *E);
  bool VisitExpressionTraitExpr(const ExpressionTraitExpr *E);
  bool VisitCXXUuidofExpr(const CXXUuidofExpr *E);
  bool VisitRequiresExpr(const RequiresExpr *E);
  bool VisitConceptSpecializationExpr(const ConceptSpecializationExpr *E);
  bool VisitCXXRewrittenBinaryOperator(const CXXRewrittenBinaryOperator *E);
  bool VisitPseudoObjectExpr(const PseudoObjectExpr *E);
  bool VisitPackIndexingExpr(const PackIndexingExpr *E);
  bool VisitRecoveryExpr(const RecoveryExpr *E);
  bool VisitAddrLabelExpr(const AddrLabelExpr *E);
  bool VisitConvertVectorExpr(const ConvertVectorExpr *E);
  bool VisitShuffleVectorExpr(const ShuffleVectorExpr *E);
  bool VisitExtVectorElementExpr(const ExtVectorElementExpr *E);
  bool VisitObjCBoxedExpr(const ObjCBoxedExpr *E);
  bool VisitCXXStdInitializerListExpr(const CXXStdInitializerListExpr *E);
  bool VisitStmtExpr(const StmtExpr *E);
  bool VisitCXXNewExpr(const CXXNewExpr *E);
  bool VisitCXXDeleteExpr(const CXXDeleteExpr *E);
  bool VisitBlockExpr(const BlockExpr *E);
  bool VisitCXXTypeidExpr(const CXXTypeidExpr *E);
  bool VisitObjCDictionaryLiteral(const ObjCDictionaryLiteral *E);
  bool VisitObjCArrayLiteral(const ObjCArrayLiteral *E);
  bool VisitDesignatedInitUpdateExpr(const DesignatedInitUpdateExpr *E);

  // Statements.
  bool visitCompoundStmt(const CompoundStmt *S);
  bool visitDeclStmt(const DeclStmt *DS, bool EvaluateConditionDecl = false);
  bool visitReturnStmt(const ReturnStmt *RS);
  bool visitIfStmt(const IfStmt *IS);
  bool visitWhileStmt(const WhileStmt *S);
  bool visitDoStmt(const DoStmt *S);
  bool visitForStmt(const ForStmt *S);
  bool visitCXXForRangeStmt(const CXXForRangeStmt *S);
  bool visitBreakStmt(const BreakStmt *S);
  bool visitContinueStmt(const ContinueStmt *S);
  bool visitSwitchStmt(const SwitchStmt *S);
  bool visitCaseStmt(const CaseStmt *S);
  bool visitDefaultStmt(const DefaultStmt *S);
  bool visitAttributedStmt(const AttributedStmt *S);
  bool visitCXXTryStmt(const CXXTryStmt *S);
  bool
  visitCXXExpansionStmtInstantiation(const CXXExpansionStmtInstantiation *S);

protected:
  bool visitStmt(const Stmt *S);
  bool visitExpr(const Expr *E, bool DestroyToplevelScope) override;
  bool visitLValueExpr(const Expr *E, bool DestroyToplevelScope) override;
  bool visitFunc(const FunctionDecl *F) override;

  bool visitDeclAndReturn(const VarDecl *VD, const Expr *Init,
                          bool ConstantContext) override;
  bool visitDtorCall(const VarDecl *VD, const APValue &Value) override;
  bool visitWithSubstitutions(const FunctionDecl *Callee,
                              ArrayRef<const Expr *> Args, const Expr *This,
                              const Expr *Condition) override;

protected:
  /// Emits scope cleanup instructions.
  bool emitCleanup();

  /// Returns a record type from a record or pointer type.
  const RecordType *getRecordTy(QualType Ty);

  /// Returns a record from a record or pointer type.
  Record *getRecord(QualType Ty);
  Record *getRecord(const RecordDecl *RD);

  /// Returns a function for the given FunctionDecl.
  /// If the function does not exist yet, it is compiled.
  const Function *getFunction(const FunctionDecl *FD);

  OptPrimType classify(const Expr *E) const { return Ctx.classify(E); }
  OptPrimType classify(QualType Ty) const { return Ctx.classify(Ty); }
  bool canClassify(const Expr *E) const { return Ctx.canClassify(E); }
  bool canClassify(QualType T) const { return Ctx.canClassify(T); }

  /// Classifies a known primitive type.
  PrimType classifyPrim(QualType Ty) const {
    if (auto T = classify(Ty)) {
      return *T;
    }
    llvm_unreachable("not a primitive type");
  }
  /// Classifies a known primitive expression.
  PrimType classifyPrim(const Expr *E) const {
    if (auto T = classify(E))
      return *T;
    llvm_unreachable("not a primitive type");
  }

  /// Evaluates an expression and places the result on the stack. If the
  /// expression is of composite type, a local variable will be created
  /// and a pointer to said variable will be placed on the stack.
  bool visit(const Expr *E) override;
  /// Compiles an initializer. This is like visit() but it will never
  /// create a variable and instead rely on a variable already having
  /// been created. visitInitializer() then relies on a pointer to this
  /// variable being on top of the stack.
  bool visitInitializer(const Expr *E);
  /// Similar, but will also pop the pointer.
  bool visitInitializerPop(const Expr *E);
  bool visitAsLValue(const Expr *E);
  /// Evaluates an expression for side effects and discards the result.
  bool discard(const Expr *E);
  /// Just pass evaluation on to \p E. This leaves all the parsing flags
  /// intact.
  bool delegate(const Expr *E);
  /// Creates and initializes a variable from the given decl.
  VarCreationState visitVarDecl(const VarDecl *VD, const Expr *Init,
                                bool Toplevel = false);
  VarCreationState visitDecl(const VarDecl *VD);
  /// Visit an APValue.
  bool visitAPValue(const APValue &Val, PrimType ValType, SourceInfo Info);
  bool visitAPValueInitializer(const APValue &Val, SourceInfo Info, QualType T,
                               bool IsCompleteClass = true);
  /// Visit the given decl as if we have a reference to it.
  bool visitDeclRef(const ValueDecl *D, const Expr *E);

  /// Visits an expression and converts it to a boolean.
  bool visitBool(const Expr *E);

  bool visitInitList(ArrayRef<const Expr *> Inits, const Expr *ArrayFiller,
                     const Expr *E);
  bool visitArrayElemInit(unsigned ElemIndex, const Expr *Init,
                          OptPrimType InitT);
  bool visitCallArgs(ArrayRef<const Expr *> Args, const FunctionDecl *FuncDecl,
                     bool Activate, bool IsOperatorCall);

  /// Creates a local primitive value.
  unsigned allocateLocalPrimitive(DeclTy &&Decl, PrimType Ty, bool IsConst,
                                  bool IsVolatile = false,
                                  ScopeKind SC = ScopeKind::Block);

  /// Allocates a space storing a local given its type.
  UnsignedOrNone allocateLocal(DeclTy &&Decl, QualType Ty = QualType(),
                               ScopeKind = ScopeKind::Block);
  UnsignedOrNone allocateTemporary(const Expr *E);

private:
  friend class VariableScope<Emitter>;
  friend class LocalScope<Emitter>;
  friend class DestructorScope<Emitter>;
  friend class DeclScope<Emitter>;
  friend class InitLinkScope<Emitter>;
  friend class InitStackScope<Emitter>;
  friend class OptionScope<Emitter>;
  friend class ArrayIndexScope<Emitter>;
  friend class SourceLocScope<Emitter>;
  friend struct InitLink;
  friend class LoopScope<Emitter>;
  friend class LabelScope<Emitter>;
  friend class SwitchScope<Emitter>;
  friend class StmtExprScope<Emitter>;
  friend class LocOverrideScope<Emitter>;

  /// Emits a zero initializer.
  bool visitZeroInitializer(PrimType T, QualType QT, const Expr *E);
  bool visitZeroRecordInitializer(const Record *R, const Expr *E,
                                  bool IsCompleteClass = true);
  bool visitZeroArrayInitializer(QualType T, const Expr *E);
  bool visitAssignment(const Expr *LHS, const Expr *RHS, const Expr *E);

  /// Emits an APSInt constant.
  bool emitConst(const llvm::APSInt &Value, PrimType Ty, SourceInfo Info);
  bool emitConst(const llvm::APInt &Value, PrimType Ty, SourceInfo Info);
  bool emitConst(const llvm::APSInt &Value, const Expr *E);
  bool emitConst(const llvm::APInt &Value, const Expr *E) {
    return emitConst(Value, classifyPrim(E), E);
  }

  /// Emits an integer constant.
  template <typename T> bool emitConst(T Value, PrimType Ty, SourceInfo Info);
  template <typename T> bool emitConst(T Value, const Expr *E);
  bool emitBool(bool V, const Expr *E) override {
    return this->emitConst(V, E);
  }

  llvm::RoundingMode getRoundingMode(const Expr *E) const {
    FPOptions FPO = E->getFPFeaturesInEffect(Ctx.getLangOpts());

    if (FPO.getRoundingMode() == llvm::RoundingMode::Dynamic)
      return llvm::RoundingMode::NearestTiesToEven;

    return FPO.getRoundingMode();
  }

  uint32_t getFPOptions(const Expr *E) const {
    return E->getFPFeaturesInEffect(Ctx.getLangOpts()).getAsOpaqueInt();
  }

  bool emitPrimCast(PrimType FromT, PrimType ToT, QualType ToQT, const Expr *E);
  bool emitIntegralCast(PrimType FromT, PrimType ToT, QualType ToQT,
                        const Expr *E);
  PrimType classifyComplexElementType(QualType T) const {
    assert(T->isAnyComplexType());

    QualType ElemType = T->getAs<ComplexType>()->getElementType();

    return *this->classify(ElemType);
  }

  PrimType classifyVectorElementType(QualType T) const {
    assert(T->isVectorType());
    return *this->classify(T->getAs<VectorType>()->getElementType());
  }

  PrimType classifyMatrixElementType(QualType T) const {
    assert(T->isMatrixType());
    return *this->classify(T->getAs<MatrixType>()->getElementType());
  }

  bool emitComplexReal(const Expr *SubExpr);
  bool emitComplexBoolCast(const Expr *E);
  bool emitComplexComparison(const Expr *LHS, const Expr *RHS,
                             const BinaryOperator *E);
  bool emitRecordDestructionPop(const Record *R, SourceInfo Loc);
  bool emitDestructionPop(const Descriptor *Desc, SourceInfo Loc);
  bool emitDummyPtr(const DeclTy &D, const Expr *E, bool CU = false);
  bool emitFloat(const APFloat &F, SourceInfo Info);
  unsigned collectBaseOffset(const QualType BaseType,
                             const QualType DerivedType);
  bool emitLambdaStaticInvokerBody(const CXXMethodDecl *MD);
  bool emitBuiltinBitCast(const CastExpr *E);

  bool emitHLSLAggregateSplat(PrimType SrcT, unsigned SrcOffset,
                              QualType DestType, const Expr *E);
  bool emitVectorConversion(const Expr *Src, const Expr *E);

  /// A scalar element extracted during HLSL aggregate flattening.
  struct HLSLFlatElement {
    unsigned LocalOffset;
    PrimType Type;
  };
  unsigned countHLSLFlatElements(QualType Ty);
  bool emitHLSLFlattenAggregate(QualType SrcType, unsigned SrcPtrOffset,
                                SmallVectorImpl<HLSLFlatElement> &Elements,
                                unsigned MaxElements, const Expr *E);
  bool emitHLSLConstructAggregate(QualType DestType,
                                  ArrayRef<HLSLFlatElement> Elements,
                                  unsigned &ElemIdx, const Expr *E);
  bool emitHLSLConstructAggregate(QualType DestType,
                                  ArrayRef<HLSLFlatElement> Elements,
                                  const Expr *E) {
    unsigned ElemIdx = 0;
    return emitHLSLConstructAggregate(DestType, Elements, ElemIdx, E);
  }

  bool compileConstructor(const CXXConstructorDecl *Ctor);
  bool compileDestructor(const CXXDestructorDecl *Dtor);
  bool compileUnionAssignmentOperator(const CXXMethodDecl *MD);

  bool checkLiteralType(const Expr *E);
  bool maybeEmitDeferredVarInit(const VarDecl *VD);

  bool refersToUnion(const Expr *E);

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

  bool SwitchInStmtExpr = false;
  bool InStmtExpr = false;
  bool ToLValue = false;

  bool VariablesAreConstexprUnknown = false;

  /// Flag inidicating if we're initializing an already created
  /// variable. This is set in visitInitializer().
  bool Initializing = false;
  const ValueDecl *InitializingDecl = nullptr;

  llvm::SmallVector<InitLink> InitStack;
  bool InitStackActive = false;

  /// Type of the expression returned by the function.
  OptPrimType ReturnType;

  /// Switch case mapping.
  CaseMap CaseLabels;
  /// Stack of label information for loops and switch statements.
  llvm::SmallVector<LabelInfo> LabelInfoStack;

  const FunctionDecl *CompilingFunction = nullptr;
};

extern template class Compiler<ByteCodeEmitter>;
extern template class Compiler<EvalEmitter>;

} // namespace interp
} // namespace clang

#endif
