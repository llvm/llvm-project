//===- FenvOpTest.cpp - Unit tests for CIR fenv operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include <gtest/gtest.h>

using namespace mlir;

namespace {

class CIRFenvOpTest : public ::testing::Test {
protected:
  CIRFenvOpTest() { context.loadDialect<cir::CIRDialect>(); }

  OwningOpRef<ModuleOp> parse(StringRef ir) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
    EXPECT_TRUE(module) << "failed to parse IR";
    return module;
  }

  template <typename OpTy> SmallVector<OpTy> findOps(ModuleOp module) {
    SmallVector<OpTy> ops;
    module.walk([&](OpTy op) { ops.push_back(op); });
    return ops;
  }

  static SmallVector<MemoryEffects::EffectInstance> getEffects(Operation *op) {
    MemoryEffectOpInterface effectsOp = cast<MemoryEffectOpInterface>(op);
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectsOp.getEffects(effects);
    return effects;
  }

  static void expectFenvReadAndWrite(Operation *op) {
    SmallVector<MemoryEffects::EffectInstance> effects = getEffects(op);
    ASSERT_EQ(effects.size(), 2u);

    unsigned reads = 0;
    unsigned writes = 0;
    for (const MemoryEffects::EffectInstance &effect : effects) {
      EXPECT_EQ(effect.getResource(),
                cir::FloatingPointEnvironmentResource::get());
      reads += isa<MemoryEffects::Read>(effect.getEffect());
      writes += isa<MemoryEffects::Write>(effect.getEffect());
    }
    EXPECT_EQ(reads, 1u);
    EXPECT_EQ(writes, 1u);
  }

  MLIRContext context;
};

TEST_F(CIRFenvOpTest, MemoryEffects) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%a: !cir.float, %b: !cir.float) {
      %0 = cir.fadd %a, %b : !cir.float
      %1 = cir.fadd %a, %b : !cir.float {fenv = #cir.fenv<>}
      %2 = cir.sqrt %a : !cir.float {fenv = #cir.fenv<>}
      %3 = cir.pow %a, %b : !cir.float {fenv = #cir.fenv<>}
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);

  SmallVector<cir::FAddOp> faddOps = findOps<cir::FAddOp>(*module);
  ASSERT_EQ(faddOps.size(), 2u);
  EXPECT_TRUE(getEffects(faddOps[0]).empty());
  EXPECT_TRUE(isMemoryEffectFree(faddOps[0]));
  expectFenvReadAndWrite(faddOps[1]);
  EXPECT_FALSE(isMemoryEffectFree(faddOps[1]));

  SmallVector<cir::SqrtOp> sqrtOps = findOps<cir::SqrtOp>(*module);
  ASSERT_EQ(sqrtOps.size(), 1u);
  expectFenvReadAndWrite(sqrtOps[0]);

  SmallVector<cir::PowOp> powOps = findOps<cir::PowOp>(*module);
  ASSERT_EQ(powOps.size(), 1u);
  expectFenvReadAndWrite(powOps[0]);
}

TEST_F(CIRFenvOpTest, Speculatability) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%a: !cir.float, %b: !cir.float) {
      %0 = cir.fadd %a, %b : !cir.float
      %1 = cir.fadd %a, %b : !cir.float {fenv = #cir.fenv<>}
      %2 = cir.fadd %a, %b : !cir.float {
        fenv = #cir.fenv<except_mode = masked>
      }
      %3 = cir.fadd %a, %b : !cir.float {
        fenv = #cir.fenv<strict_except = false>
      }
      %4 = cir.fadd %a, %b : !cir.float {
        fenv = #cir.fenv<except_mode = masked, strict_except = true>
      }
      %5 = cir.fadd %a, %b : !cir.float {
        fenv = #cir.fenv<except_mode = unmasked, strict_except = false>
      }
      %6 = cir.fadd %a, %b : !cir.float {
        fenv = #cir.fenv<except_mode = unknown, strict_except = false>
      }
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);

  SmallVector<cir::FAddOp> ops = findOps<cir::FAddOp>(*module);
  ASSERT_EQ(ops.size(), 7u);

  // Missing fenv fields use the defaults: masked exceptions and non-strict
  // exception behavior.
  for (unsigned i = 0; i != 4; ++i)
    EXPECT_TRUE(isSpeculatable(ops[i])) << "operation " << i;

  for (unsigned i = 4; i != ops.size(); ++i)
    EXPECT_FALSE(isSpeculatable(ops[i])) << "operation " << i;

  cir::FPEnvConstrainedOpInterface fenvOp =
      cast<cir::FPEnvConstrainedOpInterface>(ops[1].getOperation());
  EXPECT_EQ(fenvOp.getFenvDynamicRoundingMode(),
            cir::FPDynamicRoundingMode::Unknown);
  EXPECT_EQ(fenvOp.getFenvExceptionMode(), cir::FPExceptionMode::Masked);
  EXPECT_FALSE(fenvOp.getFenvStrictExcept());
}

} // namespace
