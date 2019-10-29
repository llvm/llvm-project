//===--- StableHash.cpp - Context to hold long-lived AST nodes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements an ABI-stable string hash based on SipHash.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StableHash.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <cstring>

using namespace clang;

#define DEBUG_TYPE "clang-stable-hash"

#define SIPHASH_ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))

#define SIPHASH_U8TO64_LE(p)                                                   \
  (((uint64_t)((p)[0])) | ((uint64_t)((p)[1]) << 8) |                          \
   ((uint64_t)((p)[2]) << 16) | ((uint64_t)((p)[3]) << 24) |                   \
   ((uint64_t)((p)[4]) << 32) | ((uint64_t)((p)[5]) << 40) |                   \
   ((uint64_t)((p)[6]) << 48) | ((uint64_t)((p)[7]) << 56))

#define SIPHASH_SIPROUND                                                       \
  do {                                                                         \
    v0 += v1;                                                                  \
    v1 = SIPHASH_ROTL(v1, 13);                                                 \
    v1 ^= v0;                                                                  \
    v0 = SIPHASH_ROTL(v0, 32);                                                 \
    v2 += v3;                                                                  \
    v3 = SIPHASH_ROTL(v3, 16);                                                 \
    v3 ^= v2;                                                                  \
    v0 += v3;                                                                  \
    v3 = SIPHASH_ROTL(v3, 21);                                                 \
    v3 ^= v0;                                                                  \
    v2 += v1;                                                                  \
    v1 = SIPHASH_ROTL(v1, 17);                                                 \
    v1 ^= v2;                                                                  \
    v2 = SIPHASH_ROTL(v2, 32);                                                 \
  } while (0)

template <int cROUNDS, int dROUNDS, class ResultTy>
static inline ResultTy siphash(const uint8_t *in, uint64_t inlen,
                               const uint8_t (&k)[16]) {
  static_assert(sizeof(ResultTy) == 8 || sizeof(ResultTy) == 16,
                "result type should be uint64_t or uint128_t");
  uint64_t v0 = 0x736f6d6570736575ULL;
  uint64_t v1 = 0x646f72616e646f6dULL;
  uint64_t v2 = 0x6c7967656e657261ULL;
  uint64_t v3 = 0x7465646279746573ULL;
  uint64_t b;
  uint64_t k0 = SIPHASH_U8TO64_LE(k);
  uint64_t k1 = SIPHASH_U8TO64_LE(k + 8);
  uint64_t m;
  int i;
  const uint8_t *end = in + inlen - (inlen % sizeof(uint64_t));
  const int left = inlen & 7;
  b = ((uint64_t)inlen) << 56;
  v3 ^= k1;
  v2 ^= k0;
  v1 ^= k1;
  v0 ^= k0;

  if (sizeof(ResultTy) == 16) {
    v1 ^= 0xee;
  }

  for (; in != end; in += 8) {
    m = SIPHASH_U8TO64_LE(in);
    v3 ^= m;

    for (i = 0; i < cROUNDS; ++i)
      SIPHASH_SIPROUND;

    v0 ^= m;
  }

  switch (left) {
  case 7:
    b |= ((uint64_t)in[6]) << 48;
    LLVM_FALLTHROUGH;
  case 6:
    b |= ((uint64_t)in[5]) << 40;
    LLVM_FALLTHROUGH;
  case 5:
    b |= ((uint64_t)in[4]) << 32;
    LLVM_FALLTHROUGH;
  case 4:
    b |= ((uint64_t)in[3]) << 24;
    LLVM_FALLTHROUGH;
  case 3:
    b |= ((uint64_t)in[2]) << 16;
    LLVM_FALLTHROUGH;
  case 2:
    b |= ((uint64_t)in[1]) << 8;
    LLVM_FALLTHROUGH;
  case 1:
    b |= ((uint64_t)in[0]);
    break;
  case 0:
    break;
  }

  v3 ^= b;

  for (i = 0; i < cROUNDS; ++i)
    SIPHASH_SIPROUND;

  v0 ^= b;

  if (sizeof(ResultTy) == 8) {
    v2 ^= 0xff;
  } else {
    v2 ^= 0xee;
  }

  for (i = 0; i < dROUNDS; ++i)
    SIPHASH_SIPROUND;

  b = v0 ^ v1 ^ v2 ^ v3;

  // This mess with the result type would be easier with 'if constexpr'.

  uint64_t firstHalf = b;
  if (sizeof(ResultTy) == 8)
    return firstHalf;

  v1 ^= 0xdd;

  for (i = 0; i < dROUNDS; ++i)
    SIPHASH_SIPROUND;

  b = v0 ^ v1 ^ v2 ^ v3;
  uint64_t secondHalf = b;

  return firstHalf
       | (ResultTy(secondHalf) << (sizeof(ResultTy) == 8 ? 0 : 64));
}

/// Compute an ABI-stable hash of the given string.
uint64_t clang::getStableStringHash(llvm::StringRef string) {
  static const uint8_t K[16] = {0xb5, 0xd4, 0xc9, 0xeb, 0x79, 0x10, 0x4a, 0x79,
                                0x6f, 0xec, 0x8b, 0x1b, 0x42, 0x87, 0x81, 0xd4};

  // The aliasing is fine here because of omnipotent char.
  auto data = reinterpret_cast<const uint8_t*>(string.data());
  return siphash<2, 4, uint64_t>(data, string.size(), K);
}

uint64_t clang::getPointerAuthStringDiscriminator(const ASTContext &ctxt,
                                                  llvm::StringRef string) {
  auto rawHash = getStableStringHash(string);

  // Don't do anything target-specific yet.

  // Produce a non-zero 16-bit discriminator.
  // We use a 16-bit discriminator because ARM64 can efficiently load
  // a 16-bit immediate into the high bits of a register without disturbing
  // the remainder of the value, which serves as a nice blend operation.
  // 16 bits is also sufficiently compact to not inflate a loader relocation.
  // We disallow zero to guarantee a different discriminator from the places
  // in the ABI that use a constant zero.
  uint64_t discriminator = (rawHash % 0xFFFF) + 1;
  LLVM_DEBUG(
    llvm::dbgs() << "Ptrauth string disc: " << llvm::utostr(discriminator)
                 << " (0x" << llvm::utohexstr(discriminator) << ")"
                 << " of: " << string << "\n");
  return discriminator;
}
