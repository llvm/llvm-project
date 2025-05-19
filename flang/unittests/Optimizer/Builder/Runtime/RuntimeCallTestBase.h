//===- RuntimeCallTestBase.cpp -- Base for runtime call generation tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_RUNTIMECALLTESTBASE_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_RUNTIMECALLTESTBASE_H

#include "gtest/gtest.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct RuntimeCallTest : public testing::Test {
public:
  void SetUp() override {
    fir::support::loadDialects(context);

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Set up a Module with a dummy function operation inside.
    // Set the insertion point in the function entry block.
    moduleOp = builder.create<mlir::ModuleOp>(loc);
    builder.setInsertionPointToStart(moduleOp->getBody());
    mlir::func::FuncOp func =
        builder.create<mlir::func::FuncOp>(loc, "runtime_unit_tests_func",
            builder.getFunctionType(std::nullopt, std::nullopt));
    auto *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    kindMap = std::make_unique<fir::KindMapping>(&context);
    firBuilder = std::make_unique<fir::FirOpBuilder>(builder, *kindMap);

    i1Ty = firBuilder->getI1Type();
    i8Ty = firBuilder->getI8Type();
    i16Ty = firBuilder->getIntegerType(16);
    i32Ty = firBuilder->getI32Type();
    i64Ty = firBuilder->getI64Type();
    i128Ty = firBuilder->getIntegerType(128);

    f32Ty = firBuilder->getF32Type();
    f64Ty = firBuilder->getF64Type();
    f80Ty = firBuilder->getF80Type();
    f128Ty = firBuilder->getF128Type();

    c4Ty = mlir::ComplexType::get(f32Ty);
    c8Ty = mlir::ComplexType::get(f64Ty);
    c10Ty = mlir::ComplexType::get(f80Ty);
    c16Ty = mlir::ComplexType::get(f128Ty);

    seqTy10 = fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
    boxTy = fir::BoxType::get(mlir::NoneType::get(firBuilder->getContext()));

    char1Ty = fir::CharacterType::getSingleton(builder.getContext(), 1);
    char2Ty = fir::CharacterType::getSingleton(builder.getContext(), 2);
    char4Ty = fir::CharacterType::getSingleton(builder.getContext(), 4);

    logical1Ty = fir::LogicalType::get(builder.getContext(), 1);
    logical2Ty = fir::LogicalType::get(builder.getContext(), 2);
    logical4Ty = fir::LogicalType::get(builder.getContext(), 4);
    logical8Ty = fir::LogicalType::get(builder.getContext(), 8);
  }

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp;
  std::unique_ptr<fir::KindMapping> kindMap;
  std::unique_ptr<fir::FirOpBuilder> firBuilder;

  // Commonly used types
  mlir::Type i1Ty;
  mlir::Type i8Ty;
  mlir::Type i16Ty;
  mlir::Type i32Ty;
  mlir::Type i64Ty;
  mlir::Type i128Ty;
  mlir::Type f32Ty;
  mlir::Type f64Ty;
  mlir::Type f80Ty;
  mlir::Type f128Ty;
  mlir::Type c4Ty;
  mlir::Type c8Ty;
  mlir::Type c10Ty;
  mlir::Type c16Ty;
  mlir::Type seqTy10;
  mlir::Type boxTy;
  mlir::Type char1Ty;
  mlir::Type char2Ty;
  mlir::Type char4Ty;
  mlir::Type logical1Ty;
  mlir::Type logical2Ty;
  mlir::Type logical4Ty;
  mlir::Type logical8Ty;
};

/// Check that the \p op is a `fir::CallOp` operation and its name matches
/// \p fctName and the number of arguments is equal to \p nbArgs.
/// Most runtime calls have two additional location arguments added. These are
/// added in this check when \p addLocArgs is true.
static inline void checkCallOp(mlir::Operation *op, llvm::StringRef fctName,
    unsigned nbArgs, bool addLocArgs = true) {
  EXPECT_TRUE(mlir::isa<fir::CallOp>(*op));
  auto callOp = mlir::dyn_cast<fir::CallOp>(*op);
  EXPECT_TRUE(callOp.getCallee().has_value());
  mlir::SymbolRefAttr callee = *callOp.getCallee();
  EXPECT_EQ(fctName, callee.getRootReference().getValue());
  // sourceFile and sourceLine are added arguments.
  if (addLocArgs)
    nbArgs += 2;
  EXPECT_EQ(nbArgs, callOp.getArgs().size());
}

/// Check the call operation from the \p result value. In some cases the
/// value is directly used in the call and sometimes there is an indirection
/// through a `fir.convert` operation. Once the `fir.call` operation is
/// retrieved the check is made by `checkCallOp`.
///
/// Directly used in `fir.call`.
/// ```
/// %result = arith.constant 1 : i32
/// %0 = fir.call @foo(%result) : (i32) -> i1
/// ```
///
/// Value used in `fir.call` through `fir.convert` indirection.
/// ```
/// %result = arith.constant 1 : i32
/// %arg = fir.convert %result : (i32) -> i16
/// %0 = fir.call @foo(%arg) : (i16) -> i1
/// ```
static inline void checkCallOpFromResultBox(mlir::Value result,
    llvm::StringRef fctName, unsigned nbArgs, bool addLocArgs = true) {
  EXPECT_TRUE(result.hasOneUse());
  const auto &u = result.user_begin();
  if (mlir::isa<fir::CallOp>(*u))
    return checkCallOp(*u, fctName, nbArgs, addLocArgs);
  auto convOp = mlir::dyn_cast<fir::ConvertOp>(*u);
  EXPECT_NE(nullptr, convOp);
  checkCallOpFromResultBox(convOp.getResult(), fctName, nbArgs, addLocArgs);
}

/// Check the operations in \p block for a `fir::CallOp` operation where the
/// function being called shares its function name with \p fctName and the
/// number of arguments is equal to \p nbArgs. Note that this check only cares
/// if the operation exists, and not the order in when the operation is called.
/// This results in exiting the test as soon as the first correct instance of
/// `fir::CallOp` is found).
static inline void checkBlockForCallOp(
    mlir::Block *block, llvm::StringRef fctName, unsigned nbArgs) {
  assert(block && "mlir::Block given is a nullptr");
  for (auto &op : block->getOperations()) {
    if (auto callOp = mlir::dyn_cast<fir::CallOp>(op)) {
      if (fctName == callOp.getCallee()->getRootReference().getValue()) {
        EXPECT_EQ(nbArgs, callOp.getArgs().size());
        return;
      }
    }
  }
  FAIL() << "No calls to " << fctName << " were found!";
}

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_RUNTIMECALLTESTBASE_H
