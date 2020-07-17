//===- OpenMPLoweringTest.cpp -- OpenMPLowering unit tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

class OpenMPLoweringTest : public testing::Test {
protected:
  void SetUp() override {
    mlir::registerDialect<mlir::omp::OpenMPDialect>();
    mlir::registerAllDialects(&ctx);
    mlirOpBuilder.reset(new mlir::OpBuilder(&ctx));
  }

  void TearDown() override { mlirOpBuilder.reset(); }

  mlir::MLIRContext ctx;
  std::unique_ptr<mlir::OpBuilder> mlirOpBuilder;
};

TEST_F(OpenMPLoweringTest, Barrier) {
  // Construct a dummy parse tree node for `!OMP barrier`.
  struct Fortran::parser::OmpSimpleStandaloneDirective barrierDirective(
      llvm::omp::Directive::OMPD_barrier);

  // Check and lower the `!OMP barrier` node to `BarrierOp` operation of
  // OpenMPDialect.
  EXPECT_EQ(barrierDirective.v, llvm::omp::Directive::OMPD_barrier);
  auto barrierOp = mlirOpBuilder->create<mlir::omp::BarrierOp>(
      mlirOpBuilder->getUnknownLoc());

  EXPECT_EQ(barrierOp.getOperationName(), "omp.barrier");
  EXPECT_EQ(succeeded(barrierOp.verify()), true);
}

TEST_F(OpenMPLoweringTest, TaskWait) {
  // Construct a dummy parse tree node for `!OMP taskwait`.
  struct Fortran::parser::OmpSimpleStandaloneDirective taskWaitDirective(
      llvm::omp::Directive::OMPD_taskwait);

  // Check and lower the `!OMP taskwait` node to `TaskwaitOp` operation of
  // OpenMPDialect.
  EXPECT_EQ(taskWaitDirective.v, llvm::omp::Directive::OMPD_taskwait);
  auto taskWaitOp = mlirOpBuilder->create<mlir::omp::TaskwaitOp>(
      mlirOpBuilder->getUnknownLoc());

  EXPECT_EQ(taskWaitOp.getOperationName(), "omp.taskwait");
  EXPECT_EQ(succeeded(taskWaitOp.verify()), true);
}

// main() from gtest_main
