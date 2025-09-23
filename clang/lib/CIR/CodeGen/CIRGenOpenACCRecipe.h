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

#include "CIRGenFunction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/Basic/OpenACCKinds.h"

#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace clang::CIRGen {
template <typename RecipeTy> class OpenACCRecipeBuilder {
  CIRGen::CIRGenFunction &cgf;
  CIRGen::CIRGenBuilderTy &builder;

  mlir::acc::ReductionOperator convertReductionOp(OpenACCReductionOperator op) {
    switch (op) {
    case OpenACCReductionOperator::Addition:
      return mlir::acc::ReductionOperator::AccAdd;
    case OpenACCReductionOperator::Multiplication:
      return mlir::acc::ReductionOperator::AccMul;
    case OpenACCReductionOperator::Max:
      return mlir::acc::ReductionOperator::AccMax;
    case OpenACCReductionOperator::Min:
      return mlir::acc::ReductionOperator::AccMin;
    case OpenACCReductionOperator::BitwiseAnd:
      return mlir::acc::ReductionOperator::AccIand;
    case OpenACCReductionOperator::BitwiseOr:
      return mlir::acc::ReductionOperator::AccIor;
    case OpenACCReductionOperator::BitwiseXOr:
      return mlir::acc::ReductionOperator::AccXor;
    case OpenACCReductionOperator::And:
      return mlir::acc::ReductionOperator::AccLand;
    case OpenACCReductionOperator::Or:
      return mlir::acc::ReductionOperator::AccLor;
    case OpenACCReductionOperator::Invalid:
      llvm_unreachable("invalid reduction operator");
    }

    llvm_unreachable("invalid reduction operator");
  }

  std::string getRecipeName(SourceRange loc, QualType baseType,
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

      MangleContext &mc = cgf.cgm.getCXXABI().getMangleContext();
      mc.mangleCanonicalTypeName(baseType, stream);
    }
    return recipeName;
  }

  void createFirstprivateRecipeCopy(
      mlir::Location loc, mlir::Location locEnd, mlir::Value mainOp,
      CIRGenFunction::AutoVarEmission tempDeclEmission,
      mlir::acc::FirstprivateRecipeOp recipe, const VarDecl *varRecipe,
      const VarDecl *temporary) {
    mlir::Block *block = builder.createBlock(
        &recipe.getCopyRegion(), recipe.getCopyRegion().end(),
        {mainOp.getType(), mainOp.getType()}, {loc, loc});
    builder.setInsertionPointToEnd(&recipe.getCopyRegion().back());
    CIRGenFunction::LexicalScope ls(cgf, loc, block);

    mlir::BlockArgument fromArg = block->getArgument(0);
    mlir::BlockArgument toArg = block->getArgument(1);

    mlir::Type elementTy =
        mlir::cast<cir::PointerType>(mainOp.getType()).getPointee();

    // Set the address of the emission to be the argument, so that we initialize
    // that instead of the variable in the other block.
    tempDeclEmission.setAllocatedAddress(
        Address{toArg, elementTy, cgf.getContext().getDeclAlign(varRecipe)});
    tempDeclEmission.EmittedAsOffload = true;

    CIRGenFunction::DeclMapRevertingRAII declMapRAII{cgf, temporary};
    cgf.setAddrOfLocalVar(
        temporary,
        Address{fromArg, elementTy, cgf.getContext().getDeclAlign(varRecipe)});

    cgf.emitAutoVarInit(tempDeclEmission);
    mlir::acc::YieldOp::create(builder, locEnd);
  }

  // Create the 'init' section of the recipe, including the 'copy' section for
  // 'firstprivate'.  Note that this function is not 'insertion point' clean, in
  // that it alters the insertion point to be inside of the 'destroy' section of
  // the recipe, but doesn't restore it aftewards.
  void createRecipeInitCopy(mlir::Location loc, mlir::Location locEnd,
                            SourceRange exprRange, mlir::Value mainOp,
                            RecipeTy recipe, const VarDecl *varRecipe,
                            const VarDecl *temporary) {
    assert(varRecipe && "Required recipe variable not set?");

    CIRGenFunction::AutoVarEmission tempDeclEmission{
        CIRGenFunction::AutoVarEmission::invalid()};
    CIRGenFunction::DeclMapRevertingRAII declMapRAII{cgf, varRecipe};

    // Do the 'init' section of the recipe IR, which does an alloca, then the
    // initialization (except for firstprivate).
    mlir::Block *block = builder.createBlock(&recipe.getInitRegion(),
                                             recipe.getInitRegion().end(),
                                             {mainOp.getType()}, {loc});
    builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
    CIRGenFunction::LexicalScope ls(cgf, loc, block);

    tempDeclEmission =
        cgf.emitAutoVarAlloca(*varRecipe, builder.saveInsertionPoint());

    // 'firstprivate' doesn't do its initialization in the 'init' section,
    // instead does it in the 'copy' section.  SO only do init here.
    // 'reduction' appears to use it too (rather than a 'copy' section), so
    // we probably have to do it here too, but we can do that when we get to
    // reduction implementation.
    if constexpr (std::is_same_v<RecipeTy, mlir::acc::PrivateRecipeOp>) {
      // We are OK with no init for builtins, arrays of builtins, or pointers,
      // else we should NYI so we know to go look for these.
      if (cgf.getContext().getLangOpts().CPlusPlus &&
          !varRecipe->getType()
               ->getPointeeOrArrayElementType()
               ->isBuiltinType() &&
          !varRecipe->getType()->isPointerType() && !varRecipe->getInit()) {
        // If we don't have any initialization recipe, we failed during Sema to
        // initialize this correctly. If we disable the
        // Sema::TentativeAnalysisScopes in SemaOpenACC::CreateInitRecipe, it'll
        // emit an error to tell us.  However, emitting those errors during
        // production is a violation of the standard, so we cannot do them.
        cgf.cgm.errorNYI(exprRange, "private default-init recipe");
      }
      cgf.emitAutoVarInit(tempDeclEmission);
    } else if constexpr (std::is_same_v<RecipeTy,
                                        mlir::acc::ReductionRecipeOp>) {
      // Unlike Private, the recipe here is always required as it has to do
      // init, not just 'default' init.
      if (!varRecipe->getInit())
        cgf.cgm.errorNYI(exprRange, "reduction init recipe");
      cgf.emitAutoVarInit(tempDeclEmission);
    }

    mlir::acc::YieldOp::create(builder, locEnd);

    if constexpr (std::is_same_v<RecipeTy, mlir::acc::FirstprivateRecipeOp>) {
      if (!varRecipe->getInit()) {
        // If we don't have any initialization recipe, we failed during Sema to
        // initialize this correctly. If we disable the
        // Sema::TentativeAnalysisScopes in SemaOpenACC::CreateInitRecipe, it'll
        // emit an error to tell us.  However, emitting those errors during
        // production is a violation of the standard, so we cannot do them.
        cgf.cgm.errorNYI(
            exprRange, "firstprivate copy-init recipe not properly generated");
      }

      createFirstprivateRecipeCopy(loc, locEnd, mainOp, tempDeclEmission,
                                   recipe, varRecipe, temporary);
    }
  }

  // This function generates the 'combiner' section for a reduction recipe. Note
  // that this function is not 'insertion point' clean, in that it alters the
  // insertion point to be inside of the 'combiner' section of the recipe, but
  // doesn't restore it aftewards.
  void createReductionRecipeCombiner(mlir::Location loc, mlir::Location locEnd,
                                     mlir::Value mainOp,
                                     mlir::acc::ReductionRecipeOp recipe) {
    mlir::Block *block = builder.createBlock(
        &recipe.getCombinerRegion(), recipe.getCombinerRegion().end(),
        {mainOp.getType(), mainOp.getType()}, {loc, loc});
    builder.setInsertionPointToEnd(&recipe.getCombinerRegion().back());
    CIRGenFunction::LexicalScope ls(cgf, loc, block);

    mlir::BlockArgument lhsArg = block->getArgument(0);

    mlir::acc::YieldOp::create(builder, locEnd, lhsArg);
  }

  // This function generates the 'destroy' section for a recipe. Note
  // that this function is not 'insertion point' clean, in that it alters the
  // insertion point to be inside of the 'destroy' section of the recipe, but
  // doesn't restore it aftewards.
  void createRecipeDestroySection(mlir::Location loc, mlir::Location locEnd,
                                  mlir::Value mainOp, CharUnits alignment,
                                  QualType baseType,
                                  mlir::Region &destroyRegion) {
    mlir::Block *block =
        builder.createBlock(&destroyRegion, destroyRegion.end(),
                            {mainOp.getType(), mainOp.getType()}, {loc, loc});
    builder.setInsertionPointToEnd(&destroyRegion.back());
    CIRGenFunction::LexicalScope ls(cgf, loc, block);

    mlir::Type elementTy =
        mlir::cast<cir::PointerType>(mainOp.getType()).getPointee();
    // The destroy region has a signature of "original item, privatized item".
    // So the 2nd item is the one that needs destroying, the former is just for
    // reference and we don't really have a need for it at the moment.
    Address addr{block->getArgument(1), elementTy, alignment};
    cgf.emitDestroy(addr, baseType,
                    cgf.getDestroyer(QualType::DK_cxx_destructor));

    mlir::acc::YieldOp::create(builder, locEnd);
  }

public:
  OpenACCRecipeBuilder(CIRGen::CIRGenFunction &cgf,
                       CIRGen::CIRGenBuilderTy &builder)
      : cgf(cgf), builder(builder) {}
  RecipeTy getOrCreateRecipe(ASTContext &astCtx, const Expr *varRef,
                             const VarDecl *varRecipe, const VarDecl *temporary,
                             OpenACCReductionOperator reductionOp,
                             DeclContext *dc, QualType baseType,
                             mlir::Value mainOp) {

    if (baseType->isPointerType() ||
        (baseType->isArrayType() && !baseType->isConstantArrayType())) {
      // It is clear that the use of pointers/VLAs in a recipe are not properly
      // generated/don't do what they are supposed to do.  In the case where we
      // have 'bounds', we can actually figure out what we want to
      // initialize/copy/destroy/compare/etc, but we haven't figured out how
      // that looks yet, both between the IR and generation code.  For now, we
      // will do an NYI error no it.
      cgf.cgm.errorNYI(
          varRef->getSourceRange(),
          "OpenACC recipe generation for pointer/non-constant arrays");
    }

    mlir::ModuleOp mod = builder.getBlock()
                             ->getParent()
                             ->template getParentOfType<mlir::ModuleOp>();

    std::string recipeName =
        getRecipeName(varRef->getSourceRange(), baseType, reductionOp);
    if (auto recipe = mod.lookupSymbol<RecipeTy>(recipeName))
      return recipe;

    mlir::Location loc = cgf.cgm.getLoc(varRef->getBeginLoc());
    mlir::Location locEnd = cgf.cgm.getLoc(varRef->getEndLoc());

    mlir::OpBuilder modBuilder(mod.getBodyRegion());
    RecipeTy recipe;

    if constexpr (std::is_same_v<RecipeTy, mlir::acc::ReductionRecipeOp>) {
      recipe = RecipeTy::create(modBuilder, loc, recipeName, mainOp.getType(),
                                convertReductionOp(reductionOp));
    } else {
      recipe = RecipeTy::create(modBuilder, loc, recipeName, mainOp.getType());
    }

    createRecipeInitCopy(loc, locEnd, varRef->getSourceRange(), mainOp, recipe,
                         varRecipe, temporary);

    if constexpr (std::is_same_v<RecipeTy, mlir::acc::ReductionRecipeOp>) {
      createReductionRecipeCombiner(loc, locEnd, mainOp, recipe);
    }

    if (varRecipe && varRecipe->needsDestruction(cgf.getContext()))
      createRecipeDestroySection(loc, locEnd, mainOp,
                                 cgf.getContext().getDeclAlign(varRecipe),
                                 baseType, recipe.getDestroyRegion());
    return recipe;
  }
};
} // namespace clang::CIRGen
