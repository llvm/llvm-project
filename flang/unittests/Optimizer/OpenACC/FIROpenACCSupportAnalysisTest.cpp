//===- FIROpenACCSupportAnalysisTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Analysis/FIROpenACCSupportAnalysis.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InternalNames.h"

using namespace mlir;

namespace {

struct FIROpenACCSupportAnalysisTest : public testing::Test {
  void SetUp() override {
    context.loadDialect<fir::FIROpsDialect, func::FuncDialect, DLTIDialect,
        LLVM::LLVMDialect>();
    kindMap = std::make_unique<fir::KindMapping>(&context);
    module = ModuleOp::create(UnknownLoc::get(&context));
    fir::setKindMapping(module, *kindMap);
    support.setImplementation(fir::acc::FIROpenACCSupportAnalysis());
  }

  std::optional<acc::TypeSizeAndAlignment> getExpectedFIRSize(Type ty) {
    std::optional<DataLayout> dl = acc::getDataLayout(module);
    if (!dl)
      return std::nullopt;
    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
        /*forceUnifiedTBAATree=*/false, *dl);
    std::optional<std::pair<uint64_t, unsigned short>> sizeAndAlignment =
        fir::getTypeSizeAndAlignment(
            UnknownLoc::get(&context), ty, *dl, typeConverter.getKindMap());
    if (!sizeAndAlignment)
      return std::nullopt;
    return acc::TypeSizeAndAlignment{
        llvm::TypeSize::getFixed(sizeAndAlignment->first),
        llvm::TypeSize::getFixed(sizeAndAlignment->second)};
  }

  /// A scalar `fir.declare` of \p uniqName. \p attrs may be null.
  Value declareScalar(StringRef uniqName, fir::FortranVariableFlagsAttr attrs) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    builder.setInsertionPointToEnd(module.getBody());
    auto func = func::FuncOp::create(builder, loc,
        "addr_" + std::to_string(nextFunc++), builder.getFunctionType({}, {}));
    Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    Value addr = fir::AllocaOp::create(builder, loc, builder.getI32Type());
    return fir::DeclareOp::create(builder, loc, addr.getType(), addr,
        /*shape=*/Value{},
        /*typeParams=*/ValueRange{}, /*dummy_scope=*/nullptr,
        /*storage=*/nullptr, /*storage_offset=*/0,
        builder.getStringAttr(uniqName), attrs,
        /*data_attr=*/cuf::DataAttributeAttr{},
        /*dummy_arg_no=*/IntegerAttr{})
        .getResult();
  }

  Value declareVariable(fir::FortranVariableFlagsAttr attrs = {}) {
    return declareScalar(fir::NameUniquer::doVariable({}, {}, 0, "x"), attrs);
  }

  MLIRContext context;
  std::unique_ptr<fir::KindMapping> kindMap;
  ModuleOp module;
  acc::OpenACCSupport support;
  unsigned nextFunc = 0;
};

// An OpenACCSupport implementation that leaves address capture to the default.
struct OpenACCSupportWithoutCapture {
  std::string getVariableName(Value, acc::VariableNameConfig) { return {}; }
  std::string getRecipeName(acc::RecipeKind, Type, Value) { return {}; }
  InFlightDiagnostic emitNYI(Location loc, const Twine &message) {
    return emitError(loc, message);
  }
};

TEST_F(FIROpenACCSupportAnalysisTest, FIRLogicalScalarSizeAndAlignment) {
  Type logicalTy = fir::LogicalType::get(&context, /*kind=*/4);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(logicalTy, module);
  std::optional<acc::TypeSizeAndAlignment> expected =
      getExpectedFIRSize(logicalTy);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, FIRCharacterScalarSizeAndAlignment) {
  Type charTy = fir::CharacterType::get(&context, /*kind=*/1, /*len=*/8);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(charTy, module);
  std::optional<acc::TypeSizeAndAlignment> expected =
      getExpectedFIRSize(charTy);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, FIRReferenceTypeUsesPointerSize) {
  Type logicalTy = fir::LogicalType::get(&context, /*kind=*/4);
  Type refTy = fir::ReferenceType::get(logicalTy);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(refTy, module);
  LLVM::LLVMPointerType ptrTy = LLVM::LLVMPointerType::get(&context);
  std::optional<acc::TypeSizeAndAlignment> expected =
      acc::getTypeSizeAndAlignment(ptrTy, module);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, BuiltinTypeDelegatesToAccUtilities) {
  Type i32 = IntegerType::get(&context, 32);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(i32, module);
  std::optional<acc::TypeSizeAndAlignment> expected =
      acc::getTypeSizeAndAlignment(i32, module);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  DataLayout dl(module);
  EXPECT_EQ(result->first, dl.getTypeSize(i32));
  EXPECT_EQ(result->second.getFixedValue(), dl.getTypeABIAlignment(i32));
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, TupleWithFIRArrayMemberSizeAndAlignment) {
  Type f32 = Float32Type::get(&context);
  Type seqTy = fir::SequenceType::get({1024}, f32);
  Type tupleTy = TupleType::get(&context, {seqTy});
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(tupleTy, module);
  std::optional<acc::TypeSizeAndAlignment> expected = getExpectedFIRSize(seqTy);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

TEST_F(FIROpenACCSupportAnalysisTest, TupleOfReferencesSizesAsPointers) {
  Type tupleTy = TupleType::get(&context,
      {fir::ReferenceType::get(IntegerType::get(&context, 32)),
          fir::ReferenceType::get(Float64Type::get(&context))});
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(tupleTy, module);
  LLVM::LLVMPointerType ptrTy = LLVM::LLVMPointerType::get(&context);
  std::optional<acc::TypeSizeAndAlignment> pointer =
      acc::getTypeSizeAndAlignment(ptrTy, module);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(pointer.has_value());
  EXPECT_EQ(result->first.getFixedValue(), pointer->first.getFixedValue() * 2);
}

// %0 = fir.alloca i32
// %1 = fir.declare %0 uniq_name("_QEx") : (!fir.ref<i32>) -> !fir.ref<i32>
// true: "_QEx" is a Fortran variable, and the declare carries none of the
// attributes that allow a callee to keep the address.
TEST_F(FIROpenACCSupportAnalysisTest, LocalVariableAddressIsNotCaptured) {
  Value var = declareVariable();
  EXPECT_TRUE(support.isAddressNotCapturedByCalls(var));
}

// %0 = fir.alloca i32
// Each line is fir.declare %0 uniq_name("_QEx") with one attribute. All false:
// fortran_attrs<target>       : the callee may retain a TARGET address
// fortran_attrs<pointer>      : the callee may retain a POINTER address
// fortran_attrs<host_assoc>   : an internal procedure reaches this through
//                               the host association tuple
// fortran_attrs<internal_assoc> : an internal procedure can access this
// fortran_attrs<asynchronous> : pending asynchronous I/O may use the address
// fortran_attrs<volatile>     : means outside the program may use the address
TEST_F(FIROpenACCSupportAnalysisTest, CapturingAttributesKeepTheAddress) {
  Value target = declareVariable(fir::FortranVariableFlagsAttr::get(
      &context, fir::FortranVariableFlagsEnum::target));
  Value pointer = declareVariable(fir::FortranVariableFlagsAttr::get(
      &context, fir::FortranVariableFlagsEnum::pointer));
  Value hostAssoc = declareVariable(fir::FortranVariableFlagsAttr::get(
      &context, fir::FortranVariableFlagsEnum::host_assoc));
  Value internalAssoc = declareVariable(fir::FortranVariableFlagsAttr::get(
      &context, fir::FortranVariableFlagsEnum::internal_assoc));
  Value asynchronous = declareVariable(fir::FortranVariableFlagsAttr::get(
      &context, fir::FortranVariableFlagsEnum::asynchronous));
  Value volatileVar = declareVariable(fir::FortranVariableFlagsAttr::get(
      &context, fir::FortranVariableFlagsEnum::fortran_volatile));

  EXPECT_FALSE(support.isAddressNotCapturedByCalls(target));
  EXPECT_FALSE(support.isAddressNotCapturedByCalls(pointer));
  EXPECT_FALSE(support.isAddressNotCapturedByCalls(hostAssoc));
  EXPECT_FALSE(support.isAddressNotCapturedByCalls(internalAssoc));
  EXPECT_FALSE(support.isAddressNotCapturedByCalls(asynchronous));
  EXPECT_FALSE(support.isAddressNotCapturedByCalls(volatileVar));
}

// %0 = fir.alloca i32
// %1 = fir.declare %0 uniq_name("_QEx") fortran_attrs<intent_in>
//        : (!fir.ref<i32>) -> !fir.ref<i32>
// true: intent(in) says the callee only reads the variable. It does not
// allow the callee to keep the address.
TEST_F(FIROpenACCSupportAnalysisTest, OtherAttributesDoNotCapture) {
  Value intentIn = declareVariable(fir::FortranVariableFlagsAttr::get(
      &context, fir::FortranVariableFlagsEnum::intent_in));
  EXPECT_TRUE(support.isAddressNotCapturedByCalls(intentIn));
}

// %a = fir.alloca i32
// false: the value is not a fir.declare, so it is not a Fortran variable.
// %0 = fir.declare %b uniq_name("_QPsub") : (!fir.ref<i32>) -> !fir.ref<i32>
// false: "_QPsub" is a procedure name, not a variable name.
// %1 = fir.declare %c uniq_name("x") : (!fir.ref<i32>) -> !fir.ref<i32>
// false: "x" is not a uniqued Fortran name, so this is a compiler temp.
TEST_F(FIROpenACCSupportAnalysisTest, NonVariableDoesNotAnswer) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  builder.setInsertionPointToEnd(module.getBody());
  auto func = func::FuncOp::create(
      builder, loc, "alloca_only", builder.getFunctionType({}, {}));
  builder.setInsertionPointToStart(func.addEntryBlock());
  Value alloca =
      fir::AllocaOp::create(builder, loc, builder.getI32Type()).getResult();
  Value procedureName =
      declareScalar(fir::NameUniquer::doProcedure({}, {}, "sub"),
          /*attrs=*/{});
  Value plainName = declareScalar("x", /*attrs=*/{});

  EXPECT_FALSE(support.isAddressNotCapturedByCalls(alloca));
  EXPECT_FALSE(support.isAddressNotCapturedByCalls(procedureName));
  EXPECT_FALSE(support.isAddressNotCapturedByCalls(plainName));
}

// %0 = fir.alloca i32
// %1 = fir.declare %0 uniq_name("_QEx") : (!fir.ref<i32>) -> !fir.ref<i32>
// false, even though the FIR implementation returns true for this declare.
// No registered implementation, and an implementation that does not define
// this method, both use the default: the address may be kept.
TEST_F(FIROpenACCSupportAnalysisTest, MissingImplementationKeepsTheAddress) {
  Value var = declareVariable();
  acc::OpenACCSupport defaultSupport;
  EXPECT_FALSE(defaultSupport.isAddressNotCapturedByCalls(var));

  acc::OpenACCSupport withoutCapture;
  withoutCapture.setImplementation(OpenACCSupportWithoutCapture());
  EXPECT_FALSE(withoutCapture.isAddressNotCapturedByCalls(var));
}

TEST_F(FIROpenACCSupportAnalysisTest, FIRBoxTypeSizeAndAlignment) {
  Type f32 = Float32Type::get(&context);
  Type seqTy = fir::SequenceType::get({4, 3}, f32);
  Type boxTy = fir::BoxType::get(seqTy);
  std::optional<acc::TypeSizeAndAlignment> result =
      support.getTypeSizeAndAlignment(boxTy, module);
  ASSERT_TRUE(result.has_value());

  std::optional<DataLayout> dl = acc::getDataLayout(module);
  ASSERT_TRUE(dl.has_value());
  fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
      /*forceUnifiedTBAATree=*/false, *dl);
  Type structTy =
      typeConverter.convertBoxTypeAsStruct(cast<fir::BaseBoxType>(boxTy));
  std::optional<acc::TypeSizeAndAlignment> expected =
      acc::getTypeSizeAndAlignment(structTy, module, *dl, &support);
  ASSERT_TRUE(expected.has_value());
  EXPECT_EQ(result->first, expected->first);
  EXPECT_EQ(result->second, expected->second);
}

} // namespace
