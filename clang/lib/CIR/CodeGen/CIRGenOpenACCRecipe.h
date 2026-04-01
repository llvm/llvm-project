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

#include "aiir/Dialect/OpenACC/OpenACC.h"

namespace clang::CIRGen {
class OpenACCRecipeBuilderBase {
  // makes the copy of the addresses of an alloca to the previous allocation.
  void makeAllocaCopy(aiir::Location loc, aiir::Type copyType,
                      aiir::Value numEltsToCopy, aiir::Value offsetPerSubarray,
                      aiir::Value destAlloca, aiir::Value srcAlloca);
  // This function generates the required alloca, similar to
  // 'emitAutoVarAlloca', except for the OpenACC array/pointer types.
  aiir::Value makeBoundsAlloca(aiir::Block *block, SourceRange exprRange,
                               aiir::Location loc, std::string_view allocaName,
                               size_t numBounds,
                               llvm::ArrayRef<QualType> boundTypes);

  void makeBoundsInit(aiir::Value alloca, aiir::Location loc,
                      aiir::Block *block, const VarDecl *allocaDecl,
                      QualType origType, bool isInitSection);

protected:
  CIRGen::CIRGenFunction &cgf;
  CIRGen::CIRGenBuilderTy &builder;

  aiir::Block *createRecipeBlock(aiir::Region &region, aiir::Type opTy,
                                 aiir::Location loc, size_t numBounds,
                                 bool isInit);
  // Creates a loop through an 'acc.bounds', leaving the 'insertion' point to be
  // the inside of the loop body. Traverses LB->UB UNLESS `inverse` is set.
  // Returns the 'subscriptedValue' changed with the new bounds subscript.
  std::pair<aiir::Value, aiir::Value>
  createBoundsLoop(aiir::Value subscriptedValue, aiir::Value subscriptedValue2,
                   aiir::Value bound, aiir::Location loc, bool inverse);

  aiir::Value createBoundsLoop(aiir::Value subscriptedValue, aiir::Value bound,
                               aiir::Location loc, bool inverse) {
    return createBoundsLoop(subscriptedValue, {}, bound, loc, inverse).first;
  }

  aiir::acc::ReductionOperator convertReductionOp(OpenACCReductionOperator op);

  // This function generates the 'combiner' section for a reduction recipe. Note
  // that this function is not 'insertion point' clean, in that it alters the
  // insertion point to be inside of the 'combiner' section of the recipe, but
  // doesn't restore it aftewards.
  void createReductionRecipeCombiner(
      aiir::Location loc, aiir::Location locEnd, aiir::Value mainOp,
      aiir::acc::ReductionRecipeOp recipe, size_t numBounds, QualType origType,
      llvm::ArrayRef<OpenACCReductionRecipe::CombinerRecipe> combinerRecipes);

  void createInitRecipe(aiir::Location loc, aiir::Location locEnd,
                        SourceRange exprRange, aiir::Value mainOp,
                        aiir::Region &recipeInitRegion, size_t numBounds,
                        llvm::ArrayRef<QualType> boundTypes,
                        const VarDecl *allocaDecl, QualType origType,
                        bool emitInitExpr);

  void createFirstprivateRecipeCopy(aiir::Location loc, aiir::Location locEnd,
                                    aiir::Value mainOp,
                                    const VarDecl *allocaDecl,
                                    const VarDecl *temporary,
                                    aiir::Region &copyRegion, size_t numBounds);

  void createRecipeDestroySection(aiir::Location loc, aiir::Location locEnd,
                                  aiir::Value mainOp, CharUnits alignment,
                                  QualType origType, size_t numBounds,
                                  QualType baseType,
                                  aiir::Region &destroyRegion);

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

      if constexpr (std::is_same_v<RecipeTy, aiir::acc::PrivateRecipeOp>) {
        stream << "privatization_";
      } else if constexpr (std::is_same_v<RecipeTy,
                                          aiir::acc::FirstprivateRecipeOp>) {
        stream << "firstprivatization_";

      } else if constexpr (std::is_same_v<RecipeTy,
                                          aiir::acc::ReductionRecipeOp>) {
        stream << "reduction_";
        // Values here are a little weird (for bitwise and/or is 'i' prefix, and
        // logical ops with 'l'), but are chosen to be the same as the AIIR
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
      ASTContext &astCtx, aiir::OpBuilder::InsertPoint &insertLocation,
      const Expr *varRef, const VarDecl *varRecipe, const VarDecl *temporary,
      OpenACCReductionOperator reductionOp, DeclContext *dc, QualType origType,
      size_t numBounds, llvm::ArrayRef<QualType> boundTypes, QualType baseType,
      aiir::Value mainOp,
      llvm::ArrayRef<OpenACCReductionRecipe::CombinerRecipe>
          reductionCombinerRecipes) {
    assert(!varRecipe->getType()->isSpecificBuiltinType(
               BuiltinType::ArraySection) &&
           "array section shouldn't make it to recipe creation");

    aiir::ModuleOp mod = builder.getBlock()
                             ->getParent()
                             ->template getParentOfType<aiir::ModuleOp>();

    std::string recipeName = getRecipeName(varRef->getSourceRange(), baseType,
                                           numBounds, reductionOp);
    if (auto recipe = mod.lookupSymbol<RecipeTy>(recipeName))
      return recipe;

    aiir::Location loc = cgf.cgm.getLoc(varRef->getBeginLoc());
    aiir::Location locEnd = cgf.cgm.getLoc(varRef->getEndLoc());

    aiir::OpBuilder modBuilder(mod.getBodyRegion());
    if (insertLocation.isSet())
      modBuilder.restoreInsertionPoint(insertLocation);
    RecipeTy recipe;

    if constexpr (std::is_same_v<RecipeTy, aiir::acc::ReductionRecipeOp>) {
      recipe = RecipeTy::create(modBuilder, loc, recipeName, mainOp.getType(),
                                convertReductionOp(reductionOp));
    } else {
      recipe = RecipeTy::create(modBuilder, loc, recipeName, mainOp.getType());
    }
    insertLocation = modBuilder.saveInsertionPoint();

    if constexpr (std::is_same_v<RecipeTy, aiir::acc::PrivateRecipeOp>) {
      createInitRecipe(loc, locEnd, varRef->getSourceRange(), mainOp,
                       recipe.getInitRegion(), numBounds, boundTypes, varRecipe,
                       origType, /*emitInitExpr=*/true);
    } else if constexpr (std::is_same_v<RecipeTy,
                                        aiir::acc::ReductionRecipeOp>) {
      createInitRecipe(loc, locEnd, varRef->getSourceRange(), mainOp,
                       recipe.getInitRegion(), numBounds, boundTypes, varRecipe,
                       origType, /*emitInitExpr=*/true);
      createReductionRecipeCombiner(loc, locEnd, mainOp, recipe, numBounds,
                                    origType, reductionCombinerRecipes);
    } else {
      static_assert(std::is_same_v<RecipeTy, aiir::acc::FirstprivateRecipeOp>);
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
