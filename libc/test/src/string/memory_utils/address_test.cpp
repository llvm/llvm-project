#include "utils/UnitTest/Test.h"
#include <src/string/memory_utils/address.h>

namespace __llvm_libc {

TEST(LlvmLibcAddress, AliasAreAddresses) {
  ASSERT_TRUE(IsAddressType<SrcAddr<1>>::Value);
  ASSERT_TRUE(IsAddressType<DstAddr<1>>::Value);
  ASSERT_TRUE(IsAddressType<NtSrcAddr<1>>::Value);
  ASSERT_TRUE(IsAddressType<NtDstAddr<1>>::Value);
}

TEST(LlvmLibcAddress, AliasHaveRightPermissions) {
  ASSERT_TRUE(SrcAddr<1>::IS_READ);
  ASSERT_TRUE(NtSrcAddr<1>::IS_READ);
  ASSERT_TRUE(DstAddr<1>::IS_WRITE);
  ASSERT_TRUE(NtDstAddr<1>::IS_WRITE);
}

TEST(LlvmLibcAddress, AliasHaveRightSemantic) {
  ASSERT_EQ(SrcAddr<1>::TEMPORALITY, Temporality::TEMPORAL);
  ASSERT_EQ(DstAddr<1>::TEMPORALITY, Temporality::TEMPORAL);
  ASSERT_EQ(NtSrcAddr<1>::TEMPORALITY, Temporality::NON_TEMPORAL);
  ASSERT_EQ(NtDstAddr<1>::TEMPORALITY, Temporality::NON_TEMPORAL);
}

TEST(LlvmLibcAddress, AliasHaveRightAlignment) {
  ASSERT_EQ(SrcAddr<1>::ALIGNMENT, size_t(1));
  ASSERT_EQ(SrcAddr<4>::ALIGNMENT, size_t(4));
}

TEST(LlvmLibcAddress, NarrowAlignment) {
  // Address 8-byte aligned, offset by 8.
  ASSERT_EQ(offsetAddr<8>(SrcAddr<8>(nullptr)).ALIGNMENT, 8UL);
  // Address 16-byte aligned, offset by 4.
  ASSERT_EQ(offsetAddr<4>(SrcAddr<16>(nullptr)).ALIGNMENT, 4UL);
  // Address 4-byte aligned, offset by 16.
  ASSERT_EQ(offsetAddr<16>(SrcAddr<4>(nullptr)).ALIGNMENT, 4UL);
  // Address 4-byte aligned, offset by 1.
  ASSERT_EQ(offsetAddr<1>(SrcAddr<4>(nullptr)).ALIGNMENT, 1UL);
  // Address 4-byte aligned, offset by 2.
  ASSERT_EQ(offsetAddr<2>(SrcAddr<4>(nullptr)).ALIGNMENT, 2UL);
  // Address 4-byte aligned, offset by 6.
  ASSERT_EQ(offsetAddr<6>(SrcAddr<4>(nullptr)).ALIGNMENT, 2UL);
  // Address 4-byte aligned, offset by 10.
  ASSERT_EQ(offsetAddr<10>(SrcAddr<4>(nullptr)).ALIGNMENT, 2UL);
  // Address 8-byte aligned, offset by 6.
  ASSERT_EQ(offsetAddr<6>(SrcAddr<8>(nullptr)).ALIGNMENT, 2UL);
}

TEST(LlvmLibcAddress, OffsetAddr) {
  ubyte a;
  SrcAddr<1> addr(&a);
  ASSERT_EQ((const void *)offsetAddr<4>(addr).ptr(), (const void *)(&a + 4));
  ASSERT_EQ((const void *)offsetAddr<32>(addr).ptr(), (const void *)(&a + 32));
}

TEST(LlvmLibcAddress, AssumeAligned) {
  SrcAddr<16> addr(nullptr);
  ASSERT_EQ(offsetAddrAssumeAligned<8>(addr, 0).ALIGNMENT, 8UL);
  ASSERT_EQ(offsetAddrAssumeAligned<1>(addr, 0).ALIGNMENT, 1UL);
  ASSERT_EQ(offsetAddrMultiplesOf<4>(addr, 0).ALIGNMENT, 4UL);
  ASSERT_EQ(offsetAddrMultiplesOf<32>(addr, 0).ALIGNMENT, 16UL);
}

TEST(LlvmLibcAddress, offsetAddrAssumeAligned) {
  ubyte a;
  SrcAddr<1> addr(&a);
  ASSERT_EQ((const void *)offsetAddrAssumeAligned<1>(addr, 17).ptr(),
            (const void *)(&a + 17));
}

TEST(LlvmLibcAddress, offsetAddrMultiplesOf) {
  ubyte a;
  SrcAddr<1> addr(&a);
  ASSERT_EQ((const void *)offsetAddrMultiplesOf<4>(addr, 16).ptr(),
            (const void *)(&a + 16));
}

} // namespace __llvm_libc
