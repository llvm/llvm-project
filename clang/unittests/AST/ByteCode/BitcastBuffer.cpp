#include "../../../lib/AST/ByteCode/BitcastBuffer.h"
#include "clang/AST/ASTContext.h"
#include "gtest/gtest.h"
#include <bitset>
#include <cassert>
#include <cmath>
#include <memory>
#include <string>

using namespace clang;
using namespace clang::interp;

TEST(BitcastBuffer, PushData) {
  BitcastBuffer Buff1(Bytes(sizeof(int)).toBits());

  const unsigned V = 0xCAFEBABE;
  Bits VSize = Bytes(sizeof(V)).toBits();
  std::byte Data[sizeof(V)];
  std::memcpy(Data, &V, sizeof(V));

  Endian HostEndianness =
      llvm::sys::IsLittleEndianHost ? Endian::Little : Endian::Big;

  Buff1.pushData(Data, Bits::zero(), VSize, HostEndianness);

  // The buffer is in host-endianness.
  if (llvm::sys::IsLittleEndianHost) {
    ASSERT_EQ(Buff1.Data[0], std::byte{0xbe});
    ASSERT_EQ(Buff1.Data[1], std::byte{0xba});
    ASSERT_EQ(Buff1.Data[2], std::byte{0xfe});
    ASSERT_EQ(Buff1.Data[3], std::byte{0xca});
  } else {
    ASSERT_EQ(Buff1.Data[0], std::byte{0xca});
    ASSERT_EQ(Buff1.Data[1], std::byte{0xfe});
    ASSERT_EQ(Buff1.Data[2], std::byte{0xba});
    ASSERT_EQ(Buff1.Data[3], std::byte{0xbe});
  }

  {
    unsigned V2;
    auto D = Buff1.copyBits(Bits::zero(), VSize, VSize, Endian::Little);
    std::memcpy(&V2, D.get(), sizeof(V));
    ASSERT_EQ(V, V2);

    D = Buff1.copyBits(Bits::zero(), VSize, VSize, Endian::Big);
    std::memcpy(&V2, D.get(), sizeof(V));
    ASSERT_EQ(V, V2);
  }

  BitcastBuffer Buff2(Bytes(sizeof(int)).toBits());
  {
    short s1 = 0xCAFE;
    short s2 = 0xBABE;
    std::byte sdata[2];

    std::memcpy(sdata, &s1, sizeof(s1));
    Buff2.pushData(sdata, Bits::zero(), Bits(sizeof(s1) * 8), HostEndianness);
    std::memcpy(sdata, &s2, sizeof(s2));
    Buff2.pushData(sdata, Bits(sizeof(s1) * 8), Bits(sizeof(s2) * 8),
                   HostEndianness);
  }

  if (llvm::sys::IsLittleEndianHost) {
    ASSERT_EQ(Buff2.Data[0], std::byte{0xfe});
    ASSERT_EQ(Buff2.Data[1], std::byte{0xca});
    ASSERT_EQ(Buff2.Data[2], std::byte{0xbe});
    ASSERT_EQ(Buff2.Data[3], std::byte{0xba});
  } else {
    ASSERT_EQ(Buff2.Data[0], std::byte{0xba});
    ASSERT_EQ(Buff2.Data[1], std::byte{0xbe});
    ASSERT_EQ(Buff2.Data[2], std::byte{0xca});
    ASSERT_EQ(Buff2.Data[3], std::byte{0xfe});
  }

  {
    unsigned V;
    auto D = Buff2.copyBits(Bits::zero(), Bits(sizeof(V) * 8),
                            Bits(sizeof(V) * 8), Endian::Little);
    std::memcpy(&V, D.get(), sizeof(V));
    ASSERT_EQ(V, 0xBABECAFE);

    D = Buff2.copyBits(Bits::zero(), Bits(sizeof(V) * 8), Bits(sizeof(V) * 8),
                       Endian::Big);
    std::memcpy(&V, D.get(), sizeof(V));
    ASSERT_EQ(V, 0xBABECAFE);
  }
}
