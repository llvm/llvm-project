//===- GenRuntimeCallsForTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass is only for developers to generate declarations/calls
/// of Fortran runtime function recognized in
/// flang/Optimizer/Transforms/RuntimeFunctions.inc table.
/// Sample of the generated FIR:
///   func.func private
///       @_FortranAioSetStatus(!fir.ref<i8>, !fir.ref<i8>, i64) ->
///       i1 attributes {fir.io, fir.runtime}
///
///   func.func @test__FortranAioSetStatus(
///       %arg0: !fir.ref<i8>, %arg1: !fir.ref<i8>, %arg2: i64) -> i1 {
///    %0 = fir.call @_FortranAioSetStatus(%arg0, %arg1, %arg2) :
///        (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
///    return %0 : i1
///  }
//===----------------------------------------------------------------------===//
#include "flang/Common/static-multimap-view.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/io-api.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

namespace fir {
#define GEN_PASS_DEF_GENRUNTIMECALLSFORTEST
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "gen-runtime-calls-for-test"

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

#define mkIOKey(X) FirmkKey(IONAME(X))
#define mkRTKey(X) FirmkKey(RTNAME(X))

namespace {
class GenRuntimeCallsForTestPass
    : public fir::impl::GenRuntimeCallsForTestBase<GenRuntimeCallsForTestPass> {
  using GenRuntimeCallsForTestBase<
      GenRuntimeCallsForTestPass>::GenRuntimeCallsForTestBase;

public:
  void runOnOperation() override;
};
} // end anonymous namespace

static constexpr llvm::StringRef testPrefix = "test_";

void GenRuntimeCallsForTestPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::OpBuilder mlirBuilder(moduleOp.getRegion());
  fir::FirOpBuilder builder(mlirBuilder, moduleOp);
  mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());

#define KNOWN_IO_FUNC(X)                                                       \
  fir::runtime::getIORuntimeFunc<mkIOKey(X)>(loc, builder)
#define KNOWN_RUNTIME_FUNC(X)                                                  \
  fir::runtime::getRuntimeFunc<mkRTKey(X)>(loc, builder)

  mlir::func::FuncOp runtimeFuncsTable[] = {
#include "flang/Optimizer/Transforms/RuntimeFunctions.inc"
  };

  if (!doGenerateCalls)
    return;

  // Generate thin wrapper functions calling the known Fortran
  // runtime functions.
  llvm::SmallVector<mlir::Operation *> newFuncs;
  for (unsigned i = 0;
       i < sizeof(runtimeFuncsTable) / sizeof(runtimeFuncsTable[0]); ++i) {
    mlir::func::FuncOp funcOp = runtimeFuncsTable[i];
    mlir::FunctionType funcTy = funcOp.getFunctionType();
    std::string name = (llvm::Twine(testPrefix) + funcOp.getName()).str();
    mlir::func::FuncOp callerFunc = builder.createFunction(loc, name, funcTy);
    callerFunc.setVisibility(mlir::SymbolTable::Visibility::Public);
    mlir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();

    // Generate the wrapper function body that consists of a call and return.
    builder.setInsertionPointToStart(callerFunc.addEntryBlock());
    mlir::Block::BlockArgListType args = callerFunc.front().getArguments();
    auto callOp = fir::CallOp::create(builder, loc, funcOp, args);
    mlir::func::ReturnOp::create(builder, loc, callOp.getResults());

    newFuncs.push_back(callerFunc.getOperation());
    builder.restoreInsertionPoint(insertPt);
  }

  // Make sure all wrapper functions are at the beginning
  // of the module.
  auto moduleBegin = moduleOp.getBody()->begin();
  for (auto func : newFuncs)
    func->moveBefore(moduleOp.getBody(), moduleBegin);
}
