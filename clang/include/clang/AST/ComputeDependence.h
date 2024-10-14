//===--- ComputeDependence.h -------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Calculate various template dependency flags for the AST.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_COMPUTEDEPENDENCE_H
#define LLVM_CLANG_AST_COMPUTEDEPENDENCE_H

#include "clang/AST/DependenceFlags.h"
#include "clang/Basic/ExceptionSpecificationType.h"
#include "clang/Support/Compiler.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {

class ASTContext;

class Expr;
class FullExpr;
class OpaqueValueExpr;
class ParenExpr;
class UnaryOperator;
class UnaryExprOrTypeTraitExpr;
class ArraySubscriptExpr;
class MatrixSubscriptExpr;
class CompoundLiteralExpr;
class ImplicitCastExpr;
class ExplicitCastExpr;
class BinaryOperator;
class ConditionalOperator;
class BinaryConditionalOperator;
class StmtExpr;
class ConvertVectorExpr;
class VAArgExpr;
class ChooseExpr;
class NoInitExpr;
class ArrayInitLoopExpr;
class ImplicitValueInitExpr;
class InitListExpr;
class ExtVectorElementExpr;
class BlockExpr;
class AsTypeExpr;
class DeclRefExpr;
class RecoveryExpr;
class CXXRewrittenBinaryOperator;
class CXXStdInitializerListExpr;
class CXXTypeidExpr;
class MSPropertyRefExpr;
class MSPropertySubscriptExpr;
class CXXUuidofExpr;
class CXXThisExpr;
class CXXThrowExpr;
class CXXBindTemporaryExpr;
class CXXScalarValueInitExpr;
class CXXDeleteExpr;
class ArrayTypeTraitExpr;
class ExpressionTraitExpr;
class CXXNoexceptExpr;
class PackExpansionExpr;
class PackIndexingExpr;
class SubstNonTypeTemplateParmExpr;
class CoroutineSuspendExpr;
class DependentCoawaitExpr;
class CXXNewExpr;
class CXXPseudoDestructorExpr;
class OverloadExpr;
class DependentScopeDeclRefExpr;
class CXXConstructExpr;
class CXXTemporaryObjectExpr;
class CXXDefaultInitExpr;
class CXXDefaultArgExpr;
class LambdaExpr;
class CXXUnresolvedConstructExpr;
class CXXDependentScopeMemberExpr;
class MaterializeTemporaryExpr;
class CXXFoldExpr;
class CXXParenListInitExpr;
class TypeTraitExpr;
class ConceptSpecializationExpr;
class SYCLUniqueStableNameExpr;
class PredefinedExpr;
class CallExpr;
class OffsetOfExpr;
class MemberExpr;
class ShuffleVectorExpr;
class GenericSelectionExpr;
class DesignatedInitExpr;
class ParenListExpr;
class PseudoObjectExpr;
class AtomicExpr;
class ArraySectionExpr;
class OMPArrayShapingExpr;
class OMPIteratorExpr;
class ObjCArrayLiteral;
class ObjCDictionaryLiteral;
class ObjCBoxedExpr;
class ObjCEncodeExpr;
class ObjCIvarRefExpr;
class ObjCPropertyRefExpr;
class ObjCSubscriptRefExpr;
class ObjCIsaExpr;
class ObjCIndirectCopyRestoreExpr;
class ObjCMessageExpr;
class OpenACCAsteriskSizeExpr;

// The following functions are called from constructors of `Expr`, so they
// should not access anything beyond basic
CLANG_ABI ExprDependence computeDependence(FullExpr *E);
CLANG_ABI ExprDependence computeDependence(OpaqueValueExpr *E);
CLANG_ABI ExprDependence computeDependence(ParenExpr *E);
CLANG_ABI ExprDependence computeDependence(UnaryOperator *E, const ASTContext &Ctx);
CLANG_ABI ExprDependence computeDependence(UnaryExprOrTypeTraitExpr *E);
CLANG_ABI ExprDependence computeDependence(ArraySubscriptExpr *E);
CLANG_ABI ExprDependence computeDependence(MatrixSubscriptExpr *E);
CLANG_ABI ExprDependence computeDependence(CompoundLiteralExpr *E);
CLANG_ABI ExprDependence computeDependence(ImplicitCastExpr *E);
CLANG_ABI ExprDependence computeDependence(ExplicitCastExpr *E);
CLANG_ABI ExprDependence computeDependence(BinaryOperator *E);
CLANG_ABI ExprDependence computeDependence(ConditionalOperator *E);
CLANG_ABI ExprDependence computeDependence(BinaryConditionalOperator *E);
CLANG_ABI ExprDependence computeDependence(StmtExpr *E, unsigned TemplateDepth);
CLANG_ABI ExprDependence computeDependence(ConvertVectorExpr *E);
CLANG_ABI ExprDependence computeDependence(VAArgExpr *E);
CLANG_ABI ExprDependence computeDependence(ChooseExpr *E);
CLANG_ABI ExprDependence computeDependence(NoInitExpr *E);
CLANG_ABI ExprDependence computeDependence(ArrayInitLoopExpr *E);
CLANG_ABI ExprDependence computeDependence(ImplicitValueInitExpr *E);
CLANG_ABI ExprDependence computeDependence(InitListExpr *E);
CLANG_ABI ExprDependence computeDependence(ExtVectorElementExpr *E);
CLANG_ABI ExprDependence computeDependence(BlockExpr *E,
                                 bool ContainsUnexpandedParameterPack);
CLANG_ABI ExprDependence computeDependence(AsTypeExpr *E);
CLANG_ABI ExprDependence computeDependence(DeclRefExpr *E, const ASTContext &Ctx);
CLANG_ABI ExprDependence computeDependence(RecoveryExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXRewrittenBinaryOperator *E);
CLANG_ABI ExprDependence computeDependence(CXXStdInitializerListExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXTypeidExpr *E);
CLANG_ABI ExprDependence computeDependence(MSPropertyRefExpr *E);
CLANG_ABI ExprDependence computeDependence(MSPropertySubscriptExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXUuidofExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXThisExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXThrowExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXBindTemporaryExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXScalarValueInitExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXDeleteExpr *E);
CLANG_ABI ExprDependence computeDependence(ArrayTypeTraitExpr *E);
CLANG_ABI ExprDependence computeDependence(ExpressionTraitExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXNoexceptExpr *E, CanThrowResult CT);
CLANG_ABI ExprDependence computeDependence(PackExpansionExpr *E);
CLANG_ABI ExprDependence computeDependence(PackIndexingExpr *E);
CLANG_ABI ExprDependence computeDependence(SubstNonTypeTemplateParmExpr *E);
CLANG_ABI ExprDependence computeDependence(CoroutineSuspendExpr *E);
CLANG_ABI ExprDependence computeDependence(DependentCoawaitExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXNewExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXPseudoDestructorExpr *E);
CLANG_ABI ExprDependence computeDependence(OverloadExpr *E, bool KnownDependent,
                                 bool KnownInstantiationDependent,
                                 bool KnownContainsUnexpandedParameterPack);
CLANG_ABI ExprDependence computeDependence(DependentScopeDeclRefExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXConstructExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXTemporaryObjectExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXDefaultInitExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXDefaultArgExpr *E);
CLANG_ABI ExprDependence computeDependence(LambdaExpr *E,
                                 bool ContainsUnexpandedParameterPack);
CLANG_ABI ExprDependence computeDependence(CXXUnresolvedConstructExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXDependentScopeMemberExpr *E);
CLANG_ABI ExprDependence computeDependence(MaterializeTemporaryExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXFoldExpr *E);
CLANG_ABI ExprDependence computeDependence(CXXParenListInitExpr *E);
CLANG_ABI ExprDependence computeDependence(TypeTraitExpr *E);
CLANG_ABI ExprDependence computeDependence(ConceptSpecializationExpr *E,
                                 bool ValueDependent);

CLANG_ABI ExprDependence computeDependence(SYCLUniqueStableNameExpr *E);
CLANG_ABI ExprDependence computeDependence(PredefinedExpr *E);
CLANG_ABI ExprDependence computeDependence(CallExpr *E, llvm::ArrayRef<Expr *> PreArgs);
CLANG_ABI ExprDependence computeDependence(OffsetOfExpr *E);
CLANG_ABI ExprDependence computeDependence(MemberExpr *E);
CLANG_ABI ExprDependence computeDependence(ShuffleVectorExpr *E);
CLANG_ABI ExprDependence computeDependence(GenericSelectionExpr *E,
                                 bool ContainsUnexpandedPack);
CLANG_ABI ExprDependence computeDependence(DesignatedInitExpr *E);
CLANG_ABI ExprDependence computeDependence(ParenListExpr *E);
CLANG_ABI ExprDependence computeDependence(PseudoObjectExpr *E);
CLANG_ABI ExprDependence computeDependence(AtomicExpr *E);

CLANG_ABI ExprDependence computeDependence(ArraySectionExpr *E);
CLANG_ABI ExprDependence computeDependence(OMPArrayShapingExpr *E);
CLANG_ABI ExprDependence computeDependence(OMPIteratorExpr *E);

CLANG_ABI ExprDependence computeDependence(ObjCArrayLiteral *E);
CLANG_ABI ExprDependence computeDependence(ObjCDictionaryLiteral *E);
CLANG_ABI ExprDependence computeDependence(ObjCBoxedExpr *E);
CLANG_ABI ExprDependence computeDependence(ObjCEncodeExpr *E);
CLANG_ABI ExprDependence computeDependence(ObjCIvarRefExpr *E);
CLANG_ABI ExprDependence computeDependence(ObjCPropertyRefExpr *E);
CLANG_ABI ExprDependence computeDependence(ObjCSubscriptRefExpr *E);
CLANG_ABI ExprDependence computeDependence(ObjCIsaExpr *E);
CLANG_ABI ExprDependence computeDependence(ObjCIndirectCopyRestoreExpr *E);
CLANG_ABI ExprDependence computeDependence(ObjCMessageExpr *E);
CLANG_ABI ExprDependence computeDependence(OpenACCAsteriskSizeExpr *E);

} // namespace clang
#endif
