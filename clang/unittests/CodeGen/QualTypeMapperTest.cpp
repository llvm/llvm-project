//===- QualTypeMapperTest.cpp - Tests for QualType to ABI type mapping ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests that QualTypeMapper maps the AArch64 SVE types onto the expected
/// LLVM ABI type representations.
///
//===----------------------------------------------------------------------===//

#include "../../lib/CodeGen/QualTypeMapper.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Testing/CommandLineArgs.h"
#include "clang/Testing/TestAST.h"
#include "llvm/ABI/Types.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

/// Parses a translation unit for an AArch64 target, which makes the SVE
/// builtin types available on the ASTContext, and exposes a QualTypeMapper
/// over the resulting ASTContext.
class QualTypeMapperSVETest : public ::testing::Test {
protected:
  QualTypeMapperSVETest()
      : AST(makeInputs()), Mapper(AST.context(), DL, Alloc) {}

  ASTContext &context() { return AST.context(); }

  /// Maps \p QT and returns it as an ABI vector type, or null if it did not
  /// map to a vector.
  const llvm::abi::VectorType *mapToVector(QualType QT) {
    return dyn_cast<llvm::abi::VectorType>(Mapper.convertType(QT));
  }

  /// Returns the underlying type of the file-scope typedef named \p Name.
  QualType lookupTypedef(StringRef Name) {
    for (Decl *D : context().getTranslationUnitDecl()->decls())
      if (const auto *TD = dyn_cast<TypedefNameDecl>(D))
        if (TD->getName() == Name)
          return TD->getUnderlyingType();
    ADD_FAILURE() << "no typedef named " << Name;
    return QualType();
  }

  /// Checks an SVE data vector of \p NF vectors of \p NumEls elements, each
  /// \p ElBits wide. \p IsFP selects whether the element is expected to be a
  /// floating-point or an integer type.
  void checkDataVector(StringRef Name, QualType QT, unsigned NumEls,
                       unsigned ElBits, unsigned NF, bool IsFP) {
    SCOPED_TRACE(Name);
    const llvm::abi::VectorType *VT = mapToVector(QT);
    ASSERT_NE(VT, nullptr) << "did not map to a vector type";

    EXPECT_EQ(VT->getVectorKind(), llvm::abi::VectorKind::SVEData);
    EXPECT_TRUE(VT->isScalable());
    EXPECT_EQ(VT->getNumElements(), llvm::ElementCount::getScalable(NumEls));
    EXPECT_EQ(VT->getNumVectors(), NF);
    EXPECT_EQ(VT->isTuple(), NF > 1);
    EXPECT_EQ(VT->getSizeInBits(),
              llvm::TypeSize::getScalable(NumEls * ElBits * NF));
    // AAPCS64 aligns every SVE data vector, including the tuples, to 16 bytes.
    EXPECT_EQ(VT->getAlignment(), llvm::Align(16));

    const llvm::abi::Type *Elt = VT->getElementType();
    EXPECT_EQ(Elt->getSizeInBits(), llvm::TypeSize::getFixed(ElBits));
    EXPECT_EQ(Elt->isFloat(), IsFP);
    EXPECT_EQ(Elt->isInteger(), !IsFP);
  }

  /// Checks an SVE predicate vector of \p NF vectors of \p NumEls elements.
  void checkPredicateVector(StringRef Name, QualType QT, unsigned NumEls,
                            unsigned NF) {
    SCOPED_TRACE(Name);
    const llvm::abi::VectorType *VT = mapToVector(QT);
    ASSERT_NE(VT, nullptr) << "did not map to a vector type";

    EXPECT_EQ(VT->getVectorKind(), llvm::abi::VectorKind::SVEPredicate);
    EXPECT_TRUE(VT->isScalable());
    EXPECT_EQ(VT->getNumElements(), llvm::ElementCount::getScalable(NumEls));
    EXPECT_EQ(VT->getNumVectors(), NF);
    EXPECT_EQ(VT->getSizeInBits(), llvm::TypeSize::getScalable(NumEls * NF));
    EXPECT_EQ(VT->getAlignment(), llvm::Align(2));

    const llvm::abi::Type *Elt = VT->getElementType();
    ASSERT_TRUE(Elt->isInteger());
    EXPECT_EQ(Elt->getSizeInBits(), llvm::TypeSize::getFixed(1));
  }

private:
  static TestInputs makeInputs() {
    TestInputs Inputs(R"c(
typedef __SVInt32_t fixed_int32_t __attribute__((arm_sve_vector_bits(256)));
typedef __SVBool_t fixed_bool_t __attribute__((arm_sve_vector_bits(256)));
typedef int generic_int32x4_t __attribute__((vector_size(16)));
)c");
    Inputs.Language = TestLanguage::Lang_C99;
    // The fixed-length SVE typedefs require a known vector length, which must
    // agree with the width in the attribute.
    Inputs.ExtraArgs = {"-triple",         "aarch64-unknown-linux-gnu",
                        "-target-feature", "+sve",
                        "-mvscale-min=2",  "-mvscale-max=2"};
    return Inputs;
  }

  TestAST AST;
  llvm::DataLayout DL;
  llvm::BumpPtrAllocator Alloc;
  CodeGen::QualTypeMapper Mapper;
};

// Every SVE data vector, including the x2/x3/x4 tuples, maps to a scalable
// vector tagged as SVE data.
TEST_F(QualTypeMapperSVETest, DataVectors) {
#define SVE_VECTOR_TYPE_INT(Name, MangledName, Id, SingletonId, NumEls,        \
                            ElBits, NF, IsSigned)                              \
  checkDataVector(#Name, context().SingletonId, NumEls, ElBits, NF,            \
                  /*IsFP=*/false);
#define SVE_VECTOR_TYPE_FLOAT(Name, MangledName, Id, SingletonId, NumEls,      \
                              ElBits, NF)                                      \
  checkDataVector(#Name, context().SingletonId, NumEls, ElBits, NF,            \
                  /*IsFP=*/true);
#define SVE_VECTOR_TYPE_BFLOAT(Name, MangledName, Id, SingletonId, NumEls,     \
                               ElBits, NF)                                     \
  checkDataVector(#Name, context().SingletonId, NumEls, ElBits, NF,            \
                  /*IsFP=*/true);
  // mfloat8 vectors use an integer element, since __mfp8 has no
  // floating-point semantics of its own.
#define SVE_VECTOR_TYPE_MFLOAT(Name, MangledName, Id, SingletonId, NumEls,     \
                               ElBits, NF)                                     \
  checkDataVector(#Name, context().SingletonId, NumEls, ElBits, NF,            \
                  /*IsFP=*/false);
#include "clang/Basic/AArch64ACLETypes.def"
}

// Every SVE predicate, including the x2/x4 tuples, maps to a scalable vector
// of one-bit elements tagged as an SVE predicate.
TEST_F(QualTypeMapperSVETest, PredicateVectors) {
#define SVE_PREDICATE_TYPE_ALL(Name, MangledName, Id, SingletonId, NumEls, NF) \
  checkPredicateVector(#Name, context().SingletonId, NumEls, NF);
#include "clang/Basic/AArch64ACLETypes.def"
}

// Spot-check a few representative types against their IR spellings, so that
// the macro-driven tests above are anchored to concrete expectations.
TEST_F(QualTypeMapperSVETest, RepresentativeTypes) {
  // svint8_t is <vscale x 16 x i8>.
  const llvm::abi::VectorType *SVInt8 = mapToVector(context().SveInt8Ty);
  ASSERT_NE(SVInt8, nullptr);
  EXPECT_EQ(SVInt8->getNumElements(), llvm::ElementCount::getScalable(16));
  EXPECT_EQ(SVInt8->getSizeInBits(), llvm::TypeSize::getScalable(128));
  EXPECT_TRUE(
      cast<llvm::abi::IntegerType>(SVInt8->getElementType())->isSigned());

  // svuint8_t has the same shape but an unsigned element.
  const llvm::abi::VectorType *SVUint8 = mapToVector(context().SveUint8Ty);
  ASSERT_NE(SVUint8, nullptr);
  EXPECT_FALSE(
      cast<llvm::abi::IntegerType>(SVUint8->getElementType())->isSigned());

  // svfloat64x2_t is two <vscale x 2 x double> vectors.
  const llvm::abi::VectorType *SVFloat64x2 =
      mapToVector(context().SveFloat64x2Ty);
  ASSERT_NE(SVFloat64x2, nullptr);
  EXPECT_EQ(SVFloat64x2->getNumVectors(), 2u);
  EXPECT_EQ(SVFloat64x2->getNumElements(), llvm::ElementCount::getScalable(2));
  EXPECT_EQ(SVFloat64x2->getSizeInBits(), llvm::TypeSize::getScalable(256));

  // svbool_t is <vscale x 16 x i1>.
  const llvm::abi::VectorType *SVBool = mapToVector(context().SveBoolTy);
  ASSERT_NE(SVBool, nullptr);
  EXPECT_EQ(SVBool->getNumElements(), llvm::ElementCount::getScalable(16));
  EXPECT_EQ(SVBool->getSizeInBits(), llvm::TypeSize::getScalable(16));
}

// __mfp8 has no floating-point semantics, so mfloat8 vectors use an 8-bit
// integer element. That makes svmfloat8_t share a representation with
// svuint8_t, which is also how the two are represented in LLVM IR.
TEST_F(QualTypeMapperSVETest, MFloat8VectorUsesIntegerElement) {
  const llvm::abi::VectorType *SVMFloat8 = mapToVector(context().SveMFloat8Ty);
  ASSERT_NE(SVMFloat8, nullptr);

  EXPECT_EQ(SVMFloat8->getVectorKind(), llvm::abi::VectorKind::SVEData);
  EXPECT_EQ(SVMFloat8->getNumElements(), llvm::ElementCount::getScalable(16));

  const auto *Elt =
      dyn_cast<llvm::abi::IntegerType>(SVMFloat8->getElementType());
  ASSERT_NE(Elt, nullptr);
  EXPECT_EQ(Elt->getSizeInBits(), llvm::TypeSize::getFixed(8));
  EXPECT_FALSE(Elt->isSigned());
}

// __SVCount_t is opaque, and is given the shape of svbool_t because it
// occupies a predicate register.
TEST_F(QualTypeMapperSVETest, SVECount) {
  const llvm::abi::VectorType *SVCount = mapToVector(context().SveCountTy);
  ASSERT_NE(SVCount, nullptr);

  EXPECT_EQ(SVCount->getVectorKind(), llvm::abi::VectorKind::SVECount);
  EXPECT_TRUE(SVCount->isSVEType());
  EXPECT_FALSE(SVCount->isSVEPredicate());
  EXPECT_TRUE(SVCount->isScalable());
  EXPECT_FALSE(SVCount->isTuple());
  EXPECT_EQ(SVCount->getNumElements(), llvm::ElementCount::getScalable(16));
  EXPECT_EQ(SVCount->getSizeInBits(), llvm::TypeSize::getScalable(16));
  EXPECT_EQ(SVCount->getAlignment(), llvm::Align(2));
}

// The arm_sve_vector_bits types are fixed-length, but they still have to be
// distinguishable from an ordinary vector of the same shape.
TEST_F(QualTypeMapperSVETest, FixedLengthSVEVectors) {
  const llvm::abi::VectorType *FixedInt32 =
      mapToVector(lookupTypedef("fixed_int32_t"));
  ASSERT_NE(FixedInt32, nullptr);
  EXPECT_EQ(FixedInt32->getVectorKind(), llvm::abi::VectorKind::SVEData);
  EXPECT_FALSE(FixedInt32->isScalable());
  EXPECT_FALSE(FixedInt32->isTuple());
  EXPECT_EQ(FixedInt32->getNumElements(), llvm::ElementCount::getFixed(8));
  EXPECT_EQ(FixedInt32->getSizeInBits(), llvm::TypeSize::getFixed(256));

  // Clang derives the element count of a fixed-length predicate by dividing
  // the vector length in bits by the square of the char width, so a 256-bit
  // vector length gives 4 elements.
  const llvm::abi::VectorType *FixedBool =
      mapToVector(lookupTypedef("fixed_bool_t"));
  ASSERT_NE(FixedBool, nullptr);
  EXPECT_EQ(FixedBool->getVectorKind(), llvm::abi::VectorKind::SVEPredicate);
  EXPECT_FALSE(FixedBool->isScalable());
  EXPECT_EQ(FixedBool->getNumElements(), llvm::ElementCount::getFixed(4));
}

// An ordinary vector must not be mistaken for an SVE type.
TEST_F(QualTypeMapperSVETest, PlainVectorIsGeneric) {
  const llvm::abi::VectorType *Int32x4 =
      mapToVector(lookupTypedef("generic_int32x4_t"));
  ASSERT_NE(Int32x4, nullptr);

  EXPECT_EQ(Int32x4->getVectorKind(), llvm::abi::VectorKind::Generic);
  EXPECT_FALSE(Int32x4->isSVEType());
  EXPECT_FALSE(Int32x4->isScalable());
  EXPECT_EQ(Int32x4->getNumElements(), llvm::ElementCount::getFixed(4));
  EXPECT_EQ(Int32x4->getSizeInBits(), llvm::TypeSize::getFixed(128));
}

} // namespace
