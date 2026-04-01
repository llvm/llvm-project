#include "flang/Optimizer/Builder/Runtime/Intrinsics.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

TEST_F(RuntimeCallTest, genGetGID) {
  aiir::Location loc = firBuilder->getUnknownLoc();
  aiir::Value result = fir::runtime::genGetGID(*firBuilder, loc);
  checkCallOp(result.getDefiningOp(), "_FortranAGetGID", /*nbArgs=*/0,
      /*addLocArgs=*/false);
}

TEST_F(RuntimeCallTest, genGetUID) {
  aiir::Location loc = firBuilder->getUnknownLoc();
  aiir::Value result = fir::runtime::genGetUID(*firBuilder, loc);
  checkCallOp(result.getDefiningOp(), "_FortranAGetUID", /*nbArgs=*/0,
      /*addLocArgs=*/false);
}
