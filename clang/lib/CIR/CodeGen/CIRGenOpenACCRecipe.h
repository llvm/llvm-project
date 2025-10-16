//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenACC clause recipes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/Basic/OpenACCKinds.h"

#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace clang::CIRGen {
class OpenACCRecipeBuilderBase {
  // makes the copy of the addresses of an alloca to the previous allocation.
  void makeAllocaCopy(mlir::Location loc, mlir::Type copyType,
                      mlir::Value numEltsToCopy, mlir::Value offsetPerSubarray,
                      mlir::Value destAlloca, mlir::Value srcAlloca);
  // This function generates the required alloca, similar to
  // 'emitAutoVarAlloca', except for the OpenACC array/pointer types.
  mlir::Value makeBoundsAlloca(mlir::Block *block, SourceRange exprRange,
                               mlir::Location loc, std::string_view allocaName,
                               size_t numBounds,
                               llvm::ArrayRef<QualType> boundTypes);

  void makeBoundsInit(mlir::Value alloca, mlir::Location loc,
                      mlir::Block *block, const VarDecl *allocaDecl,
                      QualType origType, bool isInitSection);

protected:
  CIRGen::CIRGenFunction &cgf;
  CIRGen::CIRGenBuilderTy &builder;

  mlir::Block *createRecipeBlock(mlir::Region &region, mlir::Type opTy,
                                 mlir::Location loc, size_t numBounds,
                                 bool isInit);
  // Creates a loop through an 'acc.bounds', leaving the 'insertion' point to be
  // the inside of the loop body. Traverses LB->UB UNLESS `inverse` is set.
  // Returns the 'subscriptedValue' changed with the new bounds subscript.
  std::pair<mlir::Value, mlir::Value>
  createBoundsLoop(mlir::Value subscriptedValue, mlir::Value subscriptedValue2,
                   mlir::Value bound, mlir::Location loc, bool inverse);

  mlir::Value createBoundsLoop(mlir::Value subscriptedValue, mlir::Value bound,
                               mlir::Location loc, bool inverse) {
    return createBoundsLoop(subscriptedValue, {}, bound, loc, inverse).first;
  }

  mlir::acc::ReductionOperator convertReductionOp(OpenACCReductionOperator op);

  // This function generates the 'combiner' section for a reduction recipe. Note
  // that this function is not 'insertion point' clean, in that it alters the
  // insertion point to be inside of the 'combiner' section of the recipe, but
  // doesn't restore it aftewards.
  void createReductionRecipeCombiner(
      mlir::Location loc, mlir::Location locEnd, mlir::Value mainOp,
      mlir::acc::ReductionRecipeOp recipe, size_t numBounds, QualType origType,
      llvm::ArrayRef<OpenACCReductionRecipe::CombinerRecipe> combinerRecipes);

  void createInitRecipe(mlir::Location loc, mlir::Location locEnd,
                        SourceRange exprRange, mlir::Value mainOp,
                        mlir::Region &recipeInitRegion, size_t numBounds,
                        llvm::ArrayRef<QualType> boundTypes,
                        const VarDecl *allocaDecl, QualType origType,
                        bool emitInitExpr);

  void createFirstprivateRecipeCopy(mlir::Location loc, mlir::Location locEnd,
                                    mlir::Value mainOp,
                                    const VarDecl *allocaDecl,
                                    const VarDecl *temporary,
                                    mlir::Region &copyRegion, size_t numBounds);

  void createRecipeDestroySection(mlir::Location loc, mlir::Location locEnd,
                                  mlir::Value mainOp, CharUnits alignment,
                                  QualType origType, size_t numBounds,
                                  QualType baseType,
                                  mlir::Region &destroyRegion);

  OpenACCRecipeBuilderBase(CIRGen::CIRGenFunction &cgf,
                           CIRGen::CIRGenBuilderTy &builder)
      : cgf(cgf), builder(builder) {}
};

template <typename RecipeTy>
class OpenACCRecipeBuilder : OpenACCRecipeBuilderBase {
  std::string getRecipeName(SourceRange loc, QualType baseType,
                            unsigned numBounds,
                            OpenACCReductionOperator reductionOp) {
    std::string recipeName;
    {
      llvm::raw_string_ostream stream(recipeName);

      if constexpr (std::is_same_v<RecipeTy, mlir::acc::PrivateRecipeOp>) {
        stream << "privatization_";
      } else if constexpr (std::is_same_v<RecipeTy,
                                          mlir::acc::FirstprivateRecipeOp>) {
        stream << "firstprivatization_";

      } else if constexpr (std::is_same_v<RecipeTy,
                                          mlir::acc::ReductionRecipeOp>) {
        stream << "reduction_";
        // Values here are a little weird (for bitwise and/or is 'i' prefix, and
        // logical ops with 'l'), but are chosen to be the same as the MLIR
        // dialect names as well as to match the Flang versions of these.
        switch (reductionOp) {
        case OpenACCReductionOperator::Addition:
          stream << "add_";
          break;
        case OpenACCReductionOperator::Multiplication:
          stream << "mul_";
          break;
        case OpenACCReductionOperator::Max:
          stream << "max_";
          break;
        case OpenACCReductionOperator::Min:
          stream << "min_";
          break;
        case OpenACCReductionOperator::BitwiseAnd:
          stream << "iand_";
          break;
        case OpenACCReductionOperator::BitwiseOr:
          stream << "ior_";
          break;
        case OpenACCReductionOperator::BitwiseXOr:
          stream << "xor_";
          break;
        case OpenACCReductionOperator::And:
          stream << "land_";
          break;
        case OpenACCReductionOperator::Or:
          stream << "lor_";
          break;
        case OpenACCReductionOperator::Invalid:
          llvm_unreachable("invalid reduction operator");
        }
      } else {
        static_assert(!sizeof(RecipeTy), "Unknown Recipe op kind");
      }

      //  The naming convention from Flang with bounds doesn't map to C++ types
      //  very well, so we're just going to choose our own here.
      if (numBounds)
        stream << "_Bcnt" << numBounds << '_';

      MangleContext &mc = cgf.cgm.getCXXABI().getMangleContext();
      mc.mangleCanonicalTypeName(baseType, stream);
    }
    return recipeName;
  }

public:
  OpenACCRecipeBuilder(CIRGen::CIRGenFunction &cgf,
                       CIRGen::CIRGenBuilderTy &builder)
      : OpenACCRecipeBuilderBase(cgf, builder) {}
  RecipeTy getOrCreateRecipe(
      ASTContext &astCtx, mlir::OpBuilder::InsertPoint &insertLocation,
      const Expr *varRef, const VarDecl *varRecipe, const VarDecl *temporary,
      OpenACCReductionOperator reductionOp, DeclContext *dc, QualType origType,
      size_t numBounds, llvm::ArrayRef<QualType> boundTypes, QualType baseType,
      mlir::Value mainOp,
      llvm::ArrayRef<OpenACCReductionRecipe::CombinerRecipe>
          reductionCombinerRecipes) {
    assert(!varRecipe->getType()->isSpecificBuiltinType(
               BuiltinType::ArraySection) &&
           "array section shouldn't make it to recipe creation");

    mlir::ModuleOp mod = builder.getBlock()
                             ->getParent()
                             ->template getParentOfType<mlir::ModuleOp>();

    std::string recipeName = getRecipeName(varRef->getSourceRange(), baseType,
                                           numBounds, reductionOp);
    if (auto recipe = mod.lookupSymbol<RecipeTy>(recipeName))
      return recipe;

    mlir::Location loc = cgf.cgm.getLoc(varRef->getBeginLoc());
    mlir::Location locEnd = cgf.cgm.getLoc(varRef->getEndLoc());

    mlir::OpBuilder modBuilder(mod.getBodyRegion());
    if (insertLocation.isSet())
      modBuilder.restoreInsertionPoint(insertLocation);
    RecipeTy recipe;

    if constexpr (std::is_same_v<RecipeTy, mlir::acc::ReductionRecipeOp>) {
      recipe = RecipeTy::create(modBuilder, loc, recipeName, mainOp.getType(),
                                convertReductionOp(reductionOp));
    } else {
      recipe = RecipeTy::create(modBuilder, loc, recipeName, mainOp.getType());
    }
    insertLocation = modBuilder.saveInsertionPoint();

    if constexpr (std::is_same_v<RecipeTy, mlir::acc::PrivateRecipeOp>) {
      createInitRecipe(loc, locEnd, varRef->getSourceRange(), mainOp,
                       recipe.getInitRegion(), numBounds, boundTypes, varRecipe,
                       origType, /*emitInitExpr=*/true);
    } else if constexpr (std::is_same_v<RecipeTy,
                                        mlir::acc::ReductionRecipeOp>) {
      createInitRecipe(loc, locEnd, varRef->getSourceRange(), mainOp,
                       recipe.getInitRegion(), numBounds, boundTypes, varRecipe,
                       origType, /*emitInitExpr=*/true);
      createReductionRecipeCombiner(loc, locEnd, mainOp, recipe, numBounds,
                                    origType, reductionCombinerRecipes);
    } else {
      static_assert(std::is_same_v<RecipeTy, mlir::acc::FirstprivateRecipeOp>);
      createInitRecipe(loc, locEnd, varRef->getSourceRange(), mainOp,
                       recipe.getInitRegion(), numBounds, boundTypes, varRecipe,
                       origType, /*emitInitExpr=*/false);
      createFirstprivateRecipeCopy(loc, locEnd, mainOp, varRecipe, temporary,
                                   recipe.getCopyRegion(), numBounds);
    }

    if (origType.isDestructedType())
      createRecipeDestroySection(
          loc, locEnd, mainOp, cgf.getContext().getDeclAlign(varRecipe),
          origType, numBounds, baseType, recipe.getDestroyRegion());
    return recipe;
  }
};
} // namespace clang::CIRGen
