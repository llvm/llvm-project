//===- FIROpenACCStoreLikeOpInterfaceTest.cpp - stored-value interface ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OwningOpRef.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenACC/Support/RegisterOpenACCExtensions.h"
#include "flang/Optimizer/Support/InitFIR.h"

using namespace mlir;

namespace {

// Local stand-ins for store shapes that have no production operation yet.
// fir.store is the one production implementation and is tested below.

struct AddrAndOffsetOp
    : public Op<AddrAndOffsetOp, acc::StoreLikeOpInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddrAndOffsetOp)

  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static StringRef getOperationName() { return "storelike.addr_and_offset"; }

  static void build(OpBuilder &, OperationState &state, Value addr,
      Value offset, Value dest) {
    state.addOperands({addr, offset, dest});
  }

  // %addr and %offset are both stored into %dest.
  void getStoredValues(SmallVectorImpl<acc::StoredValue> &stored) {
    stored.push_back({{acc::StoredSourceKind::Operand,
                          &getOperation()->getOpOperand(0), {}, false},
        {acc::StoreLocation{&getOperation()->getOpOperand(2)}}});
    stored.push_back({{acc::StoredSourceKind::Operand,
                          &getOperation()->getOpOperand(1), {}, false},
        {acc::StoreLocation{&getOperation()->getOpOperand(2)}}});
  }
};

struct PtrSwapOp : public Op<PtrSwapOp, acc::StoreLikeOpInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PtrSwapOp)

  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static StringRef getOperationName() { return "storelike.ptr_swap"; }

  static void build(
      OpBuilder &, OperationState &state, Value ptr1, Value ptr2) {
    state.addOperands({ptr1, ptr2});
  }

  // Each cell is written with the other cell's contents, not the pointer value.
  void getStoredValues(SmallVectorImpl<acc::StoredValue> &stored) {
    stored.push_back({{acc::StoredSourceKind::Contents,
                          &getOperation()->getOpOperand(0), {}, false},
        {acc::StoreLocation{&getOperation()->getOpOperand(1)}}});
    stored.push_back({{acc::StoredSourceKind::Contents,
                          &getOperation()->getOpOperand(1), {}, false},
        {acc::StoreLocation{&getOperation()->getOpOperand(0)}}});
  }
};

struct StoreGlobalBaseOp
    : public Op<StoreGlobalBaseOp, acc::StoreLikeOpInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StoreGlobalBaseOp)

  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static StringRef getOperationName() { return "storelike.global_base"; }

  static void build(
      OpBuilder &, OperationState &state, Value dest, SymbolRefAttr symbol) {
    state.addOperands(dest);
    state.addAttribute("symbol", symbol);
  }

  // The stored bits are the contents of @symbol.
  void getStoredValues(SmallVectorImpl<acc::StoredValue> &stored) {
    stored.push_back(
        {{acc::StoredSourceKind::Symbol, nullptr,
             getOperation()->getAttrOfType<SymbolRefAttr>("symbol"), true},
            {acc::StoreLocation{&getOperation()->getOpOperand(0)}}});
  }
};

struct StoreUnknownOp
    : public Op<StoreUnknownOp, acc::StoreLikeOpInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StoreUnknownOp)

  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static StringRef getOperationName() { return "storelike.unknown"; }

  static void build(OpBuilder &, OperationState &state, Value dest) {
    state.addOperands(dest);
  }

  // The operation stores into %dest but cannot name the source.
  void getStoredValues(SmallVectorImpl<acc::StoredValue> &stored) {
    stored.push_back({{acc::StoredSourceKind::Unknown, nullptr, {}, false},
        {acc::StoreLocation{&getOperation()->getOpOperand(0)}}});
  }
};

struct StoreLikeTestDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StoreLikeTestDialect)

  explicit StoreLikeTestDialect(MLIRContext *ctx)
      : Dialect(
            getDialectNamespace(), ctx, TypeID::get<StoreLikeTestDialect>()) {
    addOperations<AddrAndOffsetOp, PtrSwapOp, StoreGlobalBaseOp,
        StoreUnknownOp>();
  }
  static StringRef getDialectNamespace() { return "storelike"; }
};

template <typename OpTy, typename... Args>
OpTy buildOp(OpBuilder &builder, Location loc, Args &&...args) {
  OperationState state(loc, OpTy::getOperationName());
  OpTy::build(builder, state, std::forward<Args>(args)...);
  return cast<OpTy>(builder.create(state));
}

void expectStored(const acc::StoredValue &stored, acc::StoredSourceKind kind,
    OpOperand *operand, SymbolRefAttr symbol, bool dereferenced,
    OpOperand *destination) {
  EXPECT_EQ(stored.source.kind, kind);
  EXPECT_EQ(stored.source.operand, operand);
  EXPECT_EQ(stored.source.symbol, symbol);
  EXPECT_EQ(stored.source.dereferenced, dereferenced);
  ASSERT_EQ(stored.locations.size(), 1u);
  EXPECT_EQ(stored.locations[0].destination, destination);
}

struct FIROpenACCStoreLikeOpInterfaceTest : public testing::Test {
  void SetUp() override {
    DialectRegistry registry;
    fir::acc::registerOpenACCExtensions(registry);
    context.appendDialectRegistry(registry);
    fir::support::loadDialects(context);
    context.loadDialect<StoreLikeTestDialect>();
    module = ModuleOp::create(UnknownLoc::get(&context));
  }

  Block *addFunc(StringRef name, TypeRange inputs) {
    OpBuilder builder(module->getBodyRegion());
    auto func = func::FuncOp::create(
        builder, module->getLoc(), name, builder.getFunctionType(inputs, {}));
    return func.addEntryBlock();
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

TEST_F(FIROpenACCStoreLikeOpInterfaceTest, StoreOperand) {
  // fir.store %v to %p -> {{Operand, &v, {}, false}, {&p}}
  Type i32 = IntegerType::get(&context, 32);
  Block *entry = addFunc("store_operand", {i32, fir::ReferenceType::get(i32)});
  OpBuilder builder(entry, entry->begin());
  auto store = fir::StoreOp::create(
      builder, module->getLoc(), entry->getArgument(0), entry->getArgument(1));

  auto iface = dyn_cast<acc::StoreLikeOpInterface>(store.getOperation());
  ASSERT_TRUE(iface);
  SmallVector<acc::StoredValue> stored;
  iface.getStoredValues(stored);
  ASSERT_EQ(stored.size(), 1u);
  expectStored(stored[0], acc::StoredSourceKind::Operand,
      &store.getValueMutable(), {}, false, &store.getMemrefMutable());
}

TEST_F(FIROpenACCStoreLikeOpInterfaceTest, StoreAddrAndOffset) {
  // store %addr, %offset into %ptr -> two Operand entries, one destination
  Type i32 = IntegerType::get(&context, 32);
  auto refTy = fir::ReferenceType::get(i32);
  Block *entry = addFunc("store_addr_and_offset", {refTy, i32, refTy});
  OpBuilder builder(entry, entry->begin());
  auto op = buildOp<AddrAndOffsetOp>(builder, module->getLoc(),
      entry->getArgument(0), entry->getArgument(1), entry->getArgument(2));

  auto iface = dyn_cast<acc::StoreLikeOpInterface>(op.getOperation());
  ASSERT_TRUE(iface);
  SmallVector<acc::StoredValue> stored;
  iface.getStoredValues(stored);
  ASSERT_EQ(stored.size(), 2u);
  expectStored(stored[0], acc::StoredSourceKind::Operand,
      &op.getOperation()->getOpOperand(0), {}, false,
      &op.getOperation()->getOpOperand(2));
  expectStored(stored[1], acc::StoredSourceKind::Operand,
      &op.getOperation()->getOpOperand(1), {}, false,
      &op.getOperation()->getOpOperand(2));
}

TEST_F(FIROpenACCStoreLikeOpInterfaceTest, PtrSwap) {
  // ptr_swap: contents of each cell written to the other
  auto refTy = fir::ReferenceType::get(IntegerType::get(&context, 32));
  Block *entry = addFunc("ptr_swap", {refTy, refTy});
  OpBuilder builder(entry, entry->begin());
  auto op = buildOp<PtrSwapOp>(
      builder, module->getLoc(), entry->getArgument(0), entry->getArgument(1));

  auto iface = dyn_cast<acc::StoreLikeOpInterface>(op.getOperation());
  ASSERT_TRUE(iface);
  SmallVector<acc::StoredValue> stored;
  iface.getStoredValues(stored);
  ASSERT_EQ(stored.size(), 2u);
  expectStored(stored[0], acc::StoredSourceKind::Contents,
      &op.getOperation()->getOpOperand(0), {}, false,
      &op.getOperation()->getOpOperand(1));
  expectStored(stored[1], acc::StoredSourceKind::Contents,
      &op.getOperation()->getOpOperand(1), {}, false,
      &op.getOperation()->getOpOperand(0));
}

TEST_F(FIROpenACCStoreLikeOpInterfaceTest, StoreGlobalBase) {
  // store the contents of @g into %dst
  auto refTy = fir::ReferenceType::get(IntegerType::get(&context, 32));
  Block *entry = addFunc("store_global_base", {refTy});
  OpBuilder builder(entry, entry->begin());
  auto symbol = SymbolRefAttr::get(&context, "g");
  auto op = buildOp<StoreGlobalBaseOp>(
      builder, module->getLoc(), entry->getArgument(0), symbol);

  auto iface = dyn_cast<acc::StoreLikeOpInterface>(op.getOperation());
  ASSERT_TRUE(iface);
  SmallVector<acc::StoredValue> stored;
  iface.getStoredValues(stored);
  ASSERT_EQ(stored.size(), 1u);
  expectStored(stored[0], acc::StoredSourceKind::Symbol, nullptr, symbol,
      /*dereferenced=*/true, &op.getOperation()->getOpOperand(0));
}

TEST_F(FIROpenACCStoreLikeOpInterfaceTest, StoreUnknown) {
  // a runtime slot or a pointer built inside the operation
  auto refTy = fir::ReferenceType::get(IntegerType::get(&context, 32));
  Block *entry = addFunc("store_unknown", {refTy});
  OpBuilder builder(entry, entry->begin());
  auto op =
      buildOp<StoreUnknownOp>(builder, module->getLoc(), entry->getArgument(0));

  auto iface = dyn_cast<acc::StoreLikeOpInterface>(op.getOperation());
  ASSERT_TRUE(iface);
  SmallVector<acc::StoredValue> stored;
  iface.getStoredValues(stored);
  ASSERT_EQ(stored.size(), 1u);
  expectStored(stored[0], acc::StoredSourceKind::Unknown, nullptr, {}, false,
      &op.getOperation()->getOpOperand(0));
}

} // namespace
