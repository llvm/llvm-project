//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenACC clause nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/ExprCXX.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
// Simple type-trait to see if the first template arg is one of the list, so we
// can tell whether to `if-constexpr` a bunch of stuff.
template <typename ToTest, typename T, typename... Tys>
constexpr bool isOneOfTypes =
    std::is_same_v<ToTest, T> || isOneOfTypes<ToTest, Tys...>;
template <typename ToTest, typename T>
constexpr bool isOneOfTypes<ToTest, T> = std::is_same_v<ToTest, T>;

// Holds information for emitting clauses for a combined construct. We
// instantiate the clause emitter with this type so that it can use
// if-constexpr to specially handle these.
template <typename CompOpTy> struct CombinedConstructClauseInfo {
  using ComputeOpTy = CompOpTy;
  ComputeOpTy computeOp;
  mlir::acc::LoopOp loopOp;
};
template <typename ToTest> constexpr bool isCombinedType = false;
template <typename T>
constexpr bool isCombinedType<CombinedConstructClauseInfo<T>> = true;

template <typename OpTy>
class OpenACCClauseCIREmitter final
    : public OpenACCClauseVisitor<OpenACCClauseCIREmitter<OpTy>> {
  // Necessary for combined constructs.
  template <typename FriendOpTy> friend class OpenACCClauseCIREmitter;

  OpTy &operation;
  CIRGen::CIRGenFunction &cgf;
  CIRGen::CIRGenBuilderTy &builder;

  // This is necessary since a few of the clauses emit differently based on the
  // directive kind they are attached to.
  OpenACCDirectiveKind dirKind;
  // TODO(cir): This source location should be able to go away once the NYI
  // diagnostics are gone.
  SourceLocation dirLoc;

  llvm::SmallVector<mlir::acc::DeviceType> lastDeviceTypeValues;
  // Keep track of the async-clause so that we can shortcut updating the data
  // operands async clauses.
  bool hasAsyncClause = false;
  // Keep track of the data operands so that we can update their async clauses.
  llvm::SmallVector<mlir::Operation *> dataOperands;

  void clauseNotImplemented(const OpenACCClause &c) {
    cgf.cgm.errorNYI(c.getSourceRange(), "OpenACC Clause", c.getClauseKind());
  }

  void setLastDeviceTypeClause(const OpenACCDeviceTypeClause &clause) {
    lastDeviceTypeValues.clear();

    for (const DeviceTypeArgument &arg : clause.getArchitectures())
      lastDeviceTypeValues.push_back(decodeDeviceType(arg.getIdentifierInfo()));
  }

  mlir::Value emitIntExpr(const Expr *intExpr) {
    return cgf.emitOpenACCIntExpr(intExpr);
  }

  // 'condition' as an OpenACC grammar production is used for 'if' and (some
  // variants of) 'self'.  It needs to be emitted as a signless-1-bit value, so
  // this function emits the expression, then sets the unrealized conversion
  // cast correctly, and returns the completed value.
  mlir::Value createCondition(const Expr *condExpr) {
    mlir::Value condition = cgf.evaluateExprAsBool(condExpr);
    mlir::Location exprLoc = cgf.cgm.getLoc(condExpr->getBeginLoc());
    mlir::IntegerType targetType = mlir::IntegerType::get(
        &cgf.getMLIRContext(), /*width=*/1,
        mlir::IntegerType::SignednessSemantics::Signless);
    auto conversionOp = builder.create<mlir::UnrealizedConversionCastOp>(
        exprLoc, targetType, condition);
    return conversionOp.getResult(0);
  }

  mlir::Value createConstantInt(mlir::Location loc, unsigned width,
                                int64_t value) {
    return cgf.createOpenACCConstantInt(loc, width, value);
    mlir::IntegerType ty = mlir::IntegerType::get(
        &cgf.getMLIRContext(), width,
        mlir::IntegerType::SignednessSemantics::Signless);
    auto constOp = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(ty, value));

    return constOp.getResult();
  }

  mlir::Value createConstantInt(SourceLocation loc, unsigned width,
                                int64_t value) {
    return createConstantInt(cgf.cgm.getLoc(loc), width, value);
  }

  mlir::acc::DeviceType decodeDeviceType(const IdentifierInfo *ii) {
    // '*' case leaves no identifier-info, just a nullptr.
    if (!ii)
      return mlir::acc::DeviceType::Star;
    return llvm::StringSwitch<mlir::acc::DeviceType>(ii->getName())
        .CaseLower("default", mlir::acc::DeviceType::Default)
        .CaseLower("host", mlir::acc::DeviceType::Host)
        .CaseLower("multicore", mlir::acc::DeviceType::Multicore)
        .CasesLower("nvidia", "acc_device_nvidia",
                    mlir::acc::DeviceType::Nvidia)
        .CaseLower("radeon", mlir::acc::DeviceType::Radeon);
  }

  mlir::acc::GangArgType decodeGangType(OpenACCGangKind gk) {
    switch (gk) {
    case OpenACCGangKind::Num:
      return mlir::acc::GangArgType::Num;
    case OpenACCGangKind::Dim:
      return mlir::acc::GangArgType::Dim;
    case OpenACCGangKind::Static:
      return mlir::acc::GangArgType::Static;
    }
    llvm_unreachable("unknown gang kind");
  }

  template <typename U = void,
            typename = std::enable_if_t<isCombinedType<OpTy>, U>>
  void applyToLoopOp(const OpenACCClause &c) {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPoint(operation.loopOp);
    OpenACCClauseCIREmitter<mlir::acc::LoopOp> loopEmitter{
        operation.loopOp, cgf, builder, dirKind, dirLoc};
    loopEmitter.lastDeviceTypeValues = lastDeviceTypeValues;
    loopEmitter.Visit(&c);
  }

  template <typename U = void,
            typename = std::enable_if_t<isCombinedType<OpTy>, U>>
  void applyToComputeOp(const OpenACCClause &c) {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPoint(operation.computeOp);
    OpenACCClauseCIREmitter<typename OpTy::ComputeOpTy> computeEmitter{
        operation.computeOp, cgf, builder, dirKind, dirLoc};

    computeEmitter.lastDeviceTypeValues = lastDeviceTypeValues;

    // Async handler uses the first data operand to figure out where to insert
    // its information if it is present.  This ensures that the new handler will
    // correctly set the insertion point for async.
    if (!dataOperands.empty())
      computeEmitter.dataOperands.push_back(dataOperands.front());
    computeEmitter.Visit(&c);

    // Make sure all of the new data operands are kept track of here. The
    // combined constructs always apply 'async' to only the compute component,
    // so we need to collect these.
    dataOperands.append(computeEmitter.dataOperands);
  }

  mlir::acc::DataClauseModifier
  convertModifiers(OpenACCModifierKind modifiers) {
    using namespace mlir::acc;
    static_assert(static_cast<int>(OpenACCModifierKind::Zero) ==
                      static_cast<int>(DataClauseModifier::zero) &&
                  static_cast<int>(OpenACCModifierKind::Readonly) ==
                      static_cast<int>(DataClauseModifier::readonly) &&
                  static_cast<int>(OpenACCModifierKind::AlwaysIn) ==
                      static_cast<int>(DataClauseModifier::alwaysin) &&
                  static_cast<int>(OpenACCModifierKind::AlwaysOut) ==
                      static_cast<int>(DataClauseModifier::alwaysout) &&
                  static_cast<int>(OpenACCModifierKind::Capture) ==
                      static_cast<int>(DataClauseModifier::capture));

    DataClauseModifier mlirModifiers{};

    // The MLIR representation of this represents `always` as `alwaysin` +
    // `alwaysout`.  So do a small fixup here.
    if (isOpenACCModifierBitSet(modifiers, OpenACCModifierKind::Always)) {
      mlirModifiers = mlirModifiers | DataClauseModifier::always;
      modifiers &= ~OpenACCModifierKind::Always;
    }

    mlirModifiers = mlirModifiers | static_cast<DataClauseModifier>(modifiers);
    return mlirModifiers;
  }

  template <typename BeforeOpTy, typename AfterOpTy>
  void addDataOperand(const Expr *varOperand, mlir::acc::DataClause dataClause,
                      OpenACCModifierKind modifiers, bool structured,
                      bool implicit) {
    CIRGenFunction::OpenACCDataOperandInfo opInfo =
        cgf.getOpenACCDataOperandInfo(varOperand);

    auto beforeOp =
        builder.create<BeforeOpTy>(opInfo.beginLoc, opInfo.varValue, structured,
                                   implicit, opInfo.name, opInfo.bounds);
    operation.getDataClauseOperandsMutable().append(beforeOp.getResult());

    AfterOpTy afterOp;
    {
      mlir::OpBuilder::InsertionGuard guardCase(builder);
      builder.setInsertionPointAfter(operation);

      if constexpr (std::is_same_v<AfterOpTy, mlir::acc::DeleteOp> ||
                    std::is_same_v<AfterOpTy, mlir::acc::DetachOp>) {
        // Detach/Delete ops don't have the variable reference here, so they
        // take 1 fewer argument to their build function.
        afterOp = builder.create<AfterOpTy>(
            opInfo.beginLoc, beforeOp.getResult(), structured, implicit,
            opInfo.name, opInfo.bounds);
      } else {
        afterOp = builder.create<AfterOpTy>(
            opInfo.beginLoc, beforeOp.getResult(), opInfo.varValue, structured,
            implicit, opInfo.name, opInfo.bounds);
      }
    }

    // Set the 'rest' of the info for both operations.
    beforeOp.setDataClause(dataClause);
    afterOp.setDataClause(dataClause);
    beforeOp.setModifiers(convertModifiers(modifiers));
    afterOp.setModifiers(convertModifiers(modifiers));

    // Make sure we record these, so 'async' values can be updated later.
    dataOperands.push_back(beforeOp.getOperation());
    dataOperands.push_back(afterOp.getOperation());
  }

  template <typename BeforeOpTy>
  void addDataOperand(const Expr *varOperand, mlir::acc::DataClause dataClause,
                      OpenACCModifierKind modifiers, bool structured,
                      bool implicit) {
    CIRGenFunction::OpenACCDataOperandInfo opInfo =
        cgf.getOpenACCDataOperandInfo(varOperand);
    auto beforeOp =
        builder.create<BeforeOpTy>(opInfo.beginLoc, opInfo.varValue, structured,
                                   implicit, opInfo.name, opInfo.bounds);
    operation.getDataClauseOperandsMutable().append(beforeOp.getResult());

    // Set the 'rest' of the info for the operation.
    beforeOp.setDataClause(dataClause);
    beforeOp.setModifiers(convertModifiers(modifiers));

    // Make sure we record these, so 'async' values can be updated later.
    dataOperands.push_back(beforeOp.getOperation());
  }

  // Helper function that covers for the fact that we don't have this function
  // on all operation types.
  mlir::ArrayAttr getAsyncOnlyAttr() {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp,
                               mlir::acc::UpdateOp>) {
      return operation.getAsyncOnlyAttr();
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::EnterDataOp,
                                      mlir::acc::ExitDataOp>) {
      if (!operation.getAsyncAttr())
        return mlir::ArrayAttr{};

      llvm::SmallVector<mlir::Attribute> devTysTemp;
      devTysTemp.push_back(mlir::acc::DeviceTypeAttr::get(
          builder.getContext(), mlir::acc::DeviceType::None));
      return mlir::ArrayAttr::get(builder.getContext(), devTysTemp);
    } else if constexpr (isCombinedType<OpTy>) {
      return operation.computeOp.getAsyncOnlyAttr();
    }

    // Note: 'wait' has async as well, but it cannot have data clauses, so we
    // don't have to handle them here.

    llvm_unreachable("getting asyncOnly when clause not valid on operation?");
  }

  // Helper function that covers for the fact that we don't have this function
  // on all operation types.
  mlir::ArrayAttr getAsyncOperandsDeviceTypeAttr() {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp,
                               mlir::acc::UpdateOp>) {
      return operation.getAsyncOperandsDeviceTypeAttr();
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::EnterDataOp,
                                      mlir::acc::ExitDataOp>) {
      if (!operation.getAsyncOperand())
        return mlir::ArrayAttr{};

      llvm::SmallVector<mlir::Attribute> devTysTemp;
      devTysTemp.push_back(mlir::acc::DeviceTypeAttr::get(
          builder.getContext(), mlir::acc::DeviceType::None));
      return mlir::ArrayAttr::get(builder.getContext(), devTysTemp);
    } else if constexpr (isCombinedType<OpTy>) {
      return operation.computeOp.getAsyncOperandsDeviceTypeAttr();
    }

    // Note: 'wait' has async as well, but it cannot have data clauses, so we
    // don't have to handle them here.

    llvm_unreachable(
        "getting asyncOperandsDeviceType when clause not valid on operation?");
  }

  // Helper function that covers for the fact that we don't have this function
  // on all operation types.
  mlir::OperandRange getAsyncOperands() {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp,
                               mlir::acc::UpdateOp>)
      return operation.getAsyncOperands();
    else if constexpr (isOneOfTypes<OpTy, mlir::acc::EnterDataOp,
                                    mlir::acc::ExitDataOp>)
      return operation.getAsyncOperandMutable();
    else if constexpr (isCombinedType<OpTy>)
      return operation.computeOp.getAsyncOperands();

    // Note: 'wait' has async as well, but it cannot have data clauses, so we
    // don't have to handle them here.

    llvm_unreachable(
        "getting asyncOperandsDeviceType when clause not valid on operation?");
  }

  // The 'data' clauses all require that we add the 'async' values from the
  // operation to them. We've collected the data operands along the way, so use
  // that list to get the current 'async' values.
  void updateDataOperandAsyncValues() {
    if (!hasAsyncClause || dataOperands.empty())
      return;

    for (mlir::Operation *dataOp : dataOperands) {
      llvm::TypeSwitch<mlir::Operation *, void>(dataOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto op) {
            op.setAsyncOnlyAttr(getAsyncOnlyAttr());
            op.setAsyncOperandsDeviceTypeAttr(getAsyncOperandsDeviceTypeAttr());
            op.getAsyncOperandsMutable().assign(getAsyncOperands());
          })
          .Default([&](mlir::Operation *) {
            llvm_unreachable("Not a data operation?");
          });
    }
  }

  template <typename RecipeTy>
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
  template <typename RecipeTy>
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
    mlir::Block *block = builder.createBlock(
        &destroyRegion, destroyRegion.end(), {mainOp.getType()}, {loc});
    builder.setInsertionPointToEnd(&destroyRegion.back());
    CIRGenFunction::LexicalScope ls(cgf, loc, block);

    mlir::Type elementTy =
        mlir::cast<cir::PointerType>(mainOp.getType()).getPointee();
    Address addr{block->getArgument(0), elementTy, alignment};
    cgf.emitDestroy(addr, baseType,
                    cgf.getDestroyer(QualType::DK_cxx_destructor));

    mlir::acc::YieldOp::create(builder, locEnd);
  }

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

  template <typename RecipeTy>
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

    std::string recipeName = getRecipeName<RecipeTy>(varRef->getSourceRange(),
                                                     baseType, reductionOp);
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

public:
  OpenACCClauseCIREmitter(OpTy &operation, CIRGen::CIRGenFunction &cgf,
                          CIRGen::CIRGenBuilderTy &builder,
                          OpenACCDirectiveKind dirKind, SourceLocation dirLoc)
      : operation(operation), cgf(cgf), builder(builder), dirKind(dirKind),
        dirLoc(dirLoc) {}

  void VisitClause(const OpenACCClause &clause) {
    clauseNotImplemented(clause);
  }

  // The entry point for the CIR emitter. All users should use this rather than
  // 'visitClauseList', as this also handles the things that have to happen
  // 'after' the clauses are all visited.
  void emitClauses(ArrayRef<const OpenACCClause *> clauses) {
    this->VisitClauseList(clauses);
    updateDataOperandAsyncValues();
  }

  void VisitDefaultClause(const OpenACCDefaultClause &clause) {
    // This type-trait checks if 'op'(the first arg) is one of the mlir::acc
    // operations listed in the rest of the arguments.
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      switch (clause.getDefaultClauseKind()) {
      case OpenACCDefaultClauseKind::None:
        operation.setDefaultAttr(mlir::acc::ClauseDefaultValue::None);
        break;
      case OpenACCDefaultClauseKind::Present:
        operation.setDefaultAttr(mlir::acc::ClauseDefaultValue::Present);
        break;
      case OpenACCDefaultClauseKind::Invalid:
        break;
      }
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitDefaultClause");
    }
  }

  void VisitDeviceTypeClause(const OpenACCDeviceTypeClause &clause) {
    setLastDeviceTypeClause(clause);

    if constexpr (isOneOfTypes<OpTy, mlir::acc::InitOp,
                               mlir::acc::ShutdownOp>) {
      for (const DeviceTypeArgument &arg : clause.getArchitectures())
        operation.addDeviceType(builder.getContext(),
                                decodeDeviceType(arg.getIdentifierInfo()));
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::SetOp>) {
      assert(!operation.getDeviceTypeAttr() && "already have device-type?");
      assert(clause.getArchitectures().size() <= 1);

      if (!clause.getArchitectures().empty())
        operation.setDeviceType(
            decodeDeviceType(clause.getArchitectures()[0].getIdentifierInfo()));
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                                      mlir::acc::SerialOp, mlir::acc::KernelsOp,
                                      mlir::acc::DataOp, mlir::acc::LoopOp,
                                      mlir::acc::UpdateOp>) {
      // Nothing to do here, these constructs don't have any IR for these, as
      // they just modify the other clauses IR.  So setting of
      // `lastDeviceTypeValues` (done above) is all we need.
    } else if constexpr (isCombinedType<OpTy>) {
      // Nothing to do here either, combined constructs are just going to use
      // 'lastDeviceTypeValues' to set the value for the child visitor.
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. routine construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitNumWorkersClause(const OpenACCNumWorkersClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                               mlir::acc::KernelsOp>) {
      operation.addNumWorkersOperand(builder.getContext(),
                                     emitIntExpr(clause.getIntExpr()),
                                     lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitNumGangsClause");
    }
  }

  void VisitVectorLengthClause(const OpenACCVectorLengthClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                               mlir::acc::KernelsOp>) {
      operation.addVectorLengthOperand(builder.getContext(),
                                       emitIntExpr(clause.getIntExpr()),
                                       lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitVectorLengthClause");
    }
  }

  void VisitAsyncClause(const OpenACCAsyncClause &clause) {
    hasAsyncClause = true;
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp,
                               mlir::acc::EnterDataOp, mlir::acc::ExitDataOp,
                               mlir::acc::UpdateOp>) {
      if (!clause.hasIntExpr()) {
        operation.addAsyncOnly(builder.getContext(), lastDeviceTypeValues);
      } else {

        mlir::Value intExpr;
        {
          // Async int exprs can be referenced by the data operands, which means
          // that the int-exprs have to appear before them.  IF there is a data
          // operand already, set the insertion point to 'before' it.
          mlir::OpBuilder::InsertionGuard guardCase(builder);
          if (!dataOperands.empty())
            builder.setInsertionPoint(dataOperands.front());
          intExpr = emitIntExpr(clause.getIntExpr());
        }
        operation.addAsyncOperand(builder.getContext(), intExpr,
                                  lastDeviceTypeValues);
      }
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::WaitOp>) {
      // Wait doesn't have a device_type, so its handling here is slightly
      // different.
      if (!clause.hasIntExpr())
        operation.setAsync(true);
      else
        operation.getAsyncOperandMutable().append(
            emitIntExpr(clause.getIntExpr()));
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain. update construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitSelfClause(const OpenACCSelfClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp>) {
      if (clause.isEmptySelfClause()) {
        operation.setSelfAttr(true);
      } else if (clause.isConditionExprClause()) {
        assert(clause.hasConditionExpr());
        operation.getSelfCondMutable().append(
            createCondition(clause.getConditionExpr()));
      } else {
        llvm_unreachable("var-list version of self shouldn't get here");
      }
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::UpdateOp>) {
      assert(!clause.isEmptySelfClause() && !clause.isConditionExprClause() &&
             "var-list version of self required for update");
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::GetDevicePtrOp, mlir::acc::UpdateHostOp>(
            var, mlir::acc::DataClause::acc_update_self, {},
            /*structured=*/false, /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitSelfClause");
    }
  }

  void VisitHostClause(const OpenACCHostClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::UpdateOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::GetDevicePtrOp, mlir::acc::UpdateHostOp>(
            var, mlir::acc::DataClause::acc_update_host, {},
            /*structured=*/false, /*implicit=*/false);
    } else {
      llvm_unreachable("Unknown construct kind in VisitHostClause");
    }
  }

  void VisitDeviceClause(const OpenACCDeviceClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::UpdateOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::UpdateDeviceOp>(
            var, mlir::acc::DataClause::acc_update_device, {},
            /*structured=*/false, /*implicit=*/false);
    } else {
      llvm_unreachable("Unknown construct kind in VisitDeviceClause");
    }
  }

  void VisitIfClause(const OpenACCIfClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::InitOp,
                               mlir::acc::ShutdownOp, mlir::acc::SetOp,
                               mlir::acc::DataOp, mlir::acc::WaitOp,
                               mlir::acc::HostDataOp, mlir::acc::EnterDataOp,
                               mlir::acc::ExitDataOp, mlir::acc::UpdateOp>) {
      operation.getIfCondMutable().append(
          createCondition(clause.getConditionExpr()));
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitIfClause");
    }
  }

  void VisitIfPresentClause(const OpenACCIfPresentClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::HostDataOp,
                               mlir::acc::UpdateOp>) {
      operation.setIfPresent(true);
    } else {
      llvm_unreachable("unknown construct kind in VisitIfPresentClause");
    }
  }

  void VisitDeviceNumClause(const OpenACCDeviceNumClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::InitOp, mlir::acc::ShutdownOp,
                               mlir::acc::SetOp>) {
      operation.getDeviceNumMutable().append(emitIntExpr(clause.getIntExpr()));
    } else {
      llvm_unreachable(
          "init, shutdown, set, are only valid device_num constructs");
    }
  }

  void VisitNumGangsClause(const OpenACCNumGangsClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                               mlir::acc::KernelsOp>) {
      llvm::SmallVector<mlir::Value> values;
      for (const Expr *E : clause.getIntExprs())
        values.push_back(emitIntExpr(E));

      operation.addNumGangsOperands(builder.getContext(), values,
                                    lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitNumGangsClause");
    }
  }

  void VisitWaitClause(const OpenACCWaitClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp,
                               mlir::acc::EnterDataOp, mlir::acc::ExitDataOp,
                               mlir::acc::UpdateOp>) {
      if (!clause.hasExprs()) {
        operation.addWaitOnly(builder.getContext(), lastDeviceTypeValues);
      } else {
        llvm::SmallVector<mlir::Value> values;
        if (clause.hasDevNumExpr())
          values.push_back(emitIntExpr(clause.getDevNumExpr()));
        for (const Expr *E : clause.getQueueIdExprs())
          values.push_back(emitIntExpr(E));
        operation.addWaitOperands(builder.getContext(), clause.hasDevNumExpr(),
                                  values, lastDeviceTypeValues);
      }
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. update construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitDefaultAsyncClause(const OpenACCDefaultAsyncClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::SetOp>) {
      operation.getDefaultAsyncMutable().append(
          emitIntExpr(clause.getIntExpr()));
    } else {
      llvm_unreachable("set, is only valid device_num constructs");
    }
  }

  void VisitSeqClause(const OpenACCSeqClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      operation.addSeq(builder.getContext(), lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Routine construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitAutoClause(const OpenACCAutoClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      operation.addAuto(builder.getContext(), lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Routine, construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitIndependentClause(const OpenACCIndependentClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      operation.addIndependent(builder.getContext(), lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Routine construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitCollapseClause(const OpenACCCollapseClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      llvm::APInt value =
          clause.getIntExpr()->EvaluateKnownConstInt(cgf.cgm.getASTContext());

      value = value.sextOrTrunc(64);
      operation.setCollapseForDeviceTypes(builder.getContext(),
                                          lastDeviceTypeValues, value);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitCollapseClause");
    }
  }

  void VisitTileClause(const OpenACCTileClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      llvm::SmallVector<mlir::Value> values;

      for (const Expr *e : clause.getSizeExprs()) {
        mlir::Location exprLoc = cgf.cgm.getLoc(e->getBeginLoc());

        // We represent the * as -1.  Additionally, this is a constant, so we
        // can always just emit it as 64 bits to avoid having to do any more
        // work to determine signedness or size.
        if (isa<OpenACCAsteriskSizeExpr>(e)) {
          values.push_back(createConstantInt(exprLoc, 64, -1));
        } else {
          llvm::APInt curValue =
              e->EvaluateKnownConstInt(cgf.cgm.getASTContext());
          values.push_back(createConstantInt(
              exprLoc, 64, curValue.sextOrTrunc(64).getSExtValue()));
        }
      }

      operation.setTileForDeviceTypes(builder.getContext(),
                                      lastDeviceTypeValues, values);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitTileClause");
    }
  }

  void VisitWorkerClause(const OpenACCWorkerClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      if (clause.hasIntExpr())
        operation.addWorkerNumOperand(builder.getContext(),
                                      emitIntExpr(clause.getIntExpr()),
                                      lastDeviceTypeValues);
      else
        operation.addEmptyWorker(builder.getContext(), lastDeviceTypeValues);

    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitVectorClause(const OpenACCVectorClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      if (clause.hasIntExpr())
        operation.addVectorOperand(builder.getContext(),
                                   emitIntExpr(clause.getIntExpr()),
                                   lastDeviceTypeValues);
      else
        operation.addEmptyVector(builder.getContext(), lastDeviceTypeValues);

    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitGangClause(const OpenACCGangClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      if (clause.getNumExprs() == 0) {
        operation.addEmptyGang(builder.getContext(), lastDeviceTypeValues);
      } else {
        llvm::SmallVector<mlir::Value> values;
        llvm::SmallVector<mlir::acc::GangArgType> argTypes;
        for (unsigned i : llvm::index_range(0u, clause.getNumExprs())) {
          auto [kind, expr] = clause.getExpr(i);
          mlir::Location exprLoc = cgf.cgm.getLoc(expr->getBeginLoc());
          argTypes.push_back(decodeGangType(kind));
          if (kind == OpenACCGangKind::Dim) {
            llvm::APInt curValue =
                expr->EvaluateKnownConstInt(cgf.cgm.getASTContext());
            // The value is 1, 2, or 3, but the type isn't necessarily smaller
            // than 64.
            curValue = curValue.sextOrTrunc(64);
            values.push_back(
                createConstantInt(exprLoc, 64, curValue.getSExtValue()));
          } else if (isa<OpenACCAsteriskSizeExpr>(expr)) {
            values.push_back(createConstantInt(exprLoc, 64, -1));
          } else {
            values.push_back(emitIntExpr(expr));
          }
        }

        operation.addGangOperands(builder.getContext(), lastDeviceTypeValues,
                                  argTypes, values);
      }
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitGangClause");
    }
  }

  void VisitCopyClause(const OpenACCCopyClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::CopyinOp, mlir::acc::CopyoutOp>(
            var, mlir::acc::DataClause::acc_copy, clause.getModifierList(),
            /*structured=*/true,
            /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. declare construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitCopyInClause(const OpenACCCopyInClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::CopyinOp, mlir::acc::DeleteOp>(
            var, mlir::acc::DataClause::acc_copyin, clause.getModifierList(),
            /*structured=*/true,
            /*implicit=*/false);
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::EnterDataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::CopyinOp>(
            var, mlir::acc::DataClause::acc_copyin, clause.getModifierList(),
            /*structured=*/false, /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. declare construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitCopyOutClause(const OpenACCCopyOutClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::CreateOp, mlir::acc::CopyoutOp>(
            var, mlir::acc::DataClause::acc_copyout, clause.getModifierList(),
            /*structured=*/true,
            /*implicit=*/false);
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::ExitDataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::GetDevicePtrOp, mlir::acc::CopyoutOp>(
            var, mlir::acc::DataClause::acc_copyout, clause.getModifierList(),
            /*structured=*/false,
            /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. declare construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitCreateClause(const OpenACCCreateClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
            var, mlir::acc::DataClause::acc_create, clause.getModifierList(),
            /*structured=*/true,
            /*implicit=*/false);
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::EnterDataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::CreateOp>(
            var, mlir::acc::DataClause::acc_create, clause.getModifierList(),
            /*structured=*/false, /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. declare construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitDeleteClause(const OpenACCDeleteClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ExitDataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::GetDevicePtrOp, mlir::acc::DeleteOp>(
            var, mlir::acc::DataClause::acc_delete, {},
            /*structured=*/false,
            /*implicit=*/false);
    } else {
      llvm_unreachable("Unknown construct kind in VisitDeleteClause");
    }
  }

  void VisitDetachClause(const OpenACCDetachClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ExitDataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::GetDevicePtrOp, mlir::acc::DetachOp>(
            var, mlir::acc::DataClause::acc_detach, {},
            /*structured=*/false,
            /*implicit=*/false);
    } else {
      llvm_unreachable("Unknown construct kind in VisitDetachClause");
    }
  }

  void VisitFinalizeClause(const OpenACCFinalizeClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ExitDataOp>) {
      operation.setFinalize(true);
    } else {
      llvm_unreachable("Unknown construct kind in VisitFinalizeClause");
    }
  }

  void VisitUseDeviceClause(const OpenACCUseDeviceClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::HostDataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::UseDeviceOp>(
            var, mlir::acc::DataClause::acc_use_device, {}, /*structured=*/true,
            /*implicit=*/false);
    } else {
      llvm_unreachable("Unknown construct kind in VisitUseDeviceClause");
    }
  }

  void VisitDevicePtrClause(const OpenACCDevicePtrClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::DevicePtrOp>(
            var, mlir::acc::DataClause::acc_deviceptr, {},
            /*structured=*/true,
            /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. declare remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitNoCreateClause(const OpenACCNoCreateClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::NoCreateOp, mlir::acc::DeleteOp>(
            var, mlir::acc::DataClause::acc_no_create, {}, /*structured=*/true,
            /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitNoCreateClause");
    }
  }

  void VisitPresentClause(const OpenACCPresentClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::PresentOp, mlir::acc::DeleteOp>(
            var, mlir::acc::DataClause::acc_present, {}, /*structured=*/true,
            /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. declare remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitAttachClause(const OpenACCAttachClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::AttachOp, mlir::acc::DetachOp>(
            var, mlir::acc::DataClause::acc_attach, {}, /*structured=*/true,
            /*implicit=*/false);
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::EnterDataOp>) {
      for (const Expr *var : clause.getVarList())
        addDataOperand<mlir::acc::AttachOp>(
            var, mlir::acc::DataClause::acc_attach, {},
            /*structured=*/false, /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitAttachClause");
    }
  }

  void VisitPrivateClause(const OpenACCPrivateClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::LoopOp>) {
      for (const auto [varExpr, varRecipe] :
           llvm::zip_equal(clause.getVarList(), clause.getInitRecipes())) {
        CIRGenFunction::OpenACCDataOperandInfo opInfo =
            cgf.getOpenACCDataOperandInfo(varExpr);
        auto privateOp = mlir::acc::PrivateOp::create(
            builder, opInfo.beginLoc, opInfo.varValue, /*structured=*/true,
            /*implicit=*/false, opInfo.name, opInfo.bounds);
        privateOp.setDataClause(mlir::acc::DataClause::acc_private);

        {
          mlir::OpBuilder::InsertionGuard guardCase(builder);
          auto recipe = getOrCreateRecipe<mlir::acc::PrivateRecipeOp>(
              cgf.getContext(), varExpr, varRecipe, /*temporary=*/nullptr,
              OpenACCReductionOperator::Invalid,

              Decl::castToDeclContext(cgf.curFuncDecl), opInfo.baseType,
              privateOp.getResult());
          // TODO: OpenACC: The dialect is going to change in the near future to
          // have these be on a different operation, so when that changes, we
          // probably need to change these here.
          operation.addPrivatization(builder.getContext(), privateOp, recipe);
        }
      }
    } else if constexpr (isCombinedType<OpTy>) {
      // Despite this being valid on ParallelOp or SerialOp, combined type
      // applies to the 'loop'.
      applyToLoopOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitPrivateClause");
    }
  }

  void VisitFirstPrivateClause(const OpenACCFirstPrivateClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                               mlir::acc::SerialOp>) {
      for (const auto [varExpr, varRecipe] :
           llvm::zip_equal(clause.getVarList(), clause.getInitRecipes())) {
        CIRGenFunction::OpenACCDataOperandInfo opInfo =
            cgf.getOpenACCDataOperandInfo(varExpr);
        auto firstPrivateOp = mlir::acc::FirstprivateOp::create(
            builder, opInfo.beginLoc, opInfo.varValue, /*structured=*/true,
            /*implicit=*/false, opInfo.name, opInfo.bounds);

        firstPrivateOp.setDataClause(mlir::acc::DataClause::acc_firstprivate);

        {
          mlir::OpBuilder::InsertionGuard guardCase(builder);
          auto recipe = getOrCreateRecipe<mlir::acc::FirstprivateRecipeOp>(
              cgf.getContext(), varExpr, varRecipe.RecipeDecl,
              varRecipe.InitFromTemporary, OpenACCReductionOperator::Invalid,
              Decl::castToDeclContext(cgf.curFuncDecl), opInfo.baseType,
              firstPrivateOp.getResult());

          // TODO: OpenACC: The dialect is going to change in the near future to
          // have these be on a different operation, so when that changes, we
          // probably need to change these here.
          operation.addFirstPrivatization(builder.getContext(), firstPrivateOp,
                                          recipe);
        }
      }
    } else if constexpr (isCombinedType<OpTy>) {
      // Unlike 'private', 'firstprivate' applies to the compute op, not the
      // loop op.
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitFirstPrivateClause");
    }
  }

  void VisitReductionClause(const OpenACCReductionClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::LoopOp>) {
      for (const auto [varExpr, varRecipe] :
           llvm::zip_equal(clause.getVarList(), clause.getRecipes())) {
        CIRGenFunction::OpenACCDataOperandInfo opInfo =
            cgf.getOpenACCDataOperandInfo(varExpr);

        auto reductionOp = mlir::acc::ReductionOp::create(
            builder, opInfo.beginLoc, opInfo.varValue, /*structured=*/true,
            /*implicit=*/false, opInfo.name, opInfo.bounds);
        reductionOp.setDataClause(mlir::acc::DataClause::acc_reduction);

        {
          mlir::OpBuilder::InsertionGuard guardCase(builder);

          auto recipe = getOrCreateRecipe<mlir::acc::ReductionRecipeOp>(
              cgf.getContext(), varExpr, varRecipe.RecipeDecl,
              /*temporary=*/nullptr, clause.getReductionOp(),
              Decl::castToDeclContext(cgf.curFuncDecl), opInfo.baseType,
              reductionOp.getResult());

          operation.addReduction(builder.getContext(), reductionOp, recipe);
        }
      }
    } else if constexpr (isCombinedType<OpTy>) {
      // Despite this being valid on ParallelOp or SerialOp, combined type
      // applies to the 'loop'.
      applyToLoopOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitReductionClause");
    }
  }
};

template <typename OpTy>
auto makeClauseEmitter(OpTy &op, CIRGen::CIRGenFunction &cgf,
                       CIRGen::CIRGenBuilderTy &builder,
                       OpenACCDirectiveKind dirKind, SourceLocation dirLoc) {
  return OpenACCClauseCIREmitter<OpTy>(op, cgf, builder, dirKind, dirLoc);
}
} // namespace

template <typename Op>
void CIRGenFunction::emitOpenACCClauses(
    Op &op, OpenACCDirectiveKind dirKind, SourceLocation dirLoc,
    ArrayRef<const OpenACCClause *> clauses) {
  mlir::OpBuilder::InsertionGuard guardCase(builder);

  // Sets insertion point before the 'op', since every new expression needs to
  // be before the operation.
  builder.setInsertionPoint(op);
  makeClauseEmitter(op, *this, builder, dirKind, dirLoc).emitClauses(clauses);
}

#define EXPL_SPEC(N)                                                           \
  template void CIRGenFunction::emitOpenACCClauses<N>(                         \
      N &, OpenACCDirectiveKind, SourceLocation,                               \
      ArrayRef<const OpenACCClause *>);
EXPL_SPEC(mlir::acc::ParallelOp)
EXPL_SPEC(mlir::acc::SerialOp)
EXPL_SPEC(mlir::acc::KernelsOp)
EXPL_SPEC(mlir::acc::LoopOp)
EXPL_SPEC(mlir::acc::DataOp)
EXPL_SPEC(mlir::acc::InitOp)
EXPL_SPEC(mlir::acc::ShutdownOp)
EXPL_SPEC(mlir::acc::SetOp)
EXPL_SPEC(mlir::acc::WaitOp)
EXPL_SPEC(mlir::acc::HostDataOp)
EXPL_SPEC(mlir::acc::EnterDataOp)
EXPL_SPEC(mlir::acc::ExitDataOp)
EXPL_SPEC(mlir::acc::UpdateOp)
#undef EXPL_SPEC

template <typename ComputeOp, typename LoopOp>
void CIRGenFunction::emitOpenACCClauses(
    ComputeOp &op, LoopOp &loopOp, OpenACCDirectiveKind dirKind,
    SourceLocation dirLoc, ArrayRef<const OpenACCClause *> clauses) {
  static_assert(std::is_same_v<mlir::acc::LoopOp, LoopOp>);

  CombinedConstructClauseInfo<ComputeOp> inf{op, loopOp};
  // We cannot set the insertion point here and do so in the emitter, but make
  // sure we reset it with the 'guard' anyway.
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  makeClauseEmitter(inf, *this, builder, dirKind, dirLoc).emitClauses(clauses);
}

#define EXPL_SPEC(N)                                                           \
  template void CIRGenFunction::emitOpenACCClauses<N, mlir::acc::LoopOp>(      \
      N &, mlir::acc::LoopOp &, OpenACCDirectiveKind, SourceLocation,          \
      ArrayRef<const OpenACCClause *>);

EXPL_SPEC(mlir::acc::ParallelOp)
EXPL_SPEC(mlir::acc::SerialOp)
EXPL_SPEC(mlir::acc::KernelsOp)
#undef EXPL_SPEC
