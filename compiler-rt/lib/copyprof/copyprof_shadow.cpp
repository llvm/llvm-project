//===-- copyprof_shadow.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the shadow memory management for CopyProf, tracking
/// the copy status and modification state of application memory.
///
//===----------------------------------------------------------------------===//

#include "copyprof_shadow.h"

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __copyprof {
namespace {

constexpr uptr kBitsPerShadowByte = 8;
static_assert(
    1 << kShadowScale == kBitsPerShadowByte,
    "CopyProf tracks 1 bit per application byte (8 bits per shadow byte)");

// Tracks whether application memory is marked as a copy.
ShadowMemory g_copy_shadow;

// Returns a mask with bits [`start_bit`, `end_bit`) set.
unsigned char BitMask(uptr start_bit, uptr end_bit) {
  CHECK_LE(start_bit, end_bit);
  return static_cast<unsigned char>((1 << end_bit) - (1 << start_bit));
}

// Whether bits [`start_bit`, `end_bit`) of `shadow_byte` are all set.
bool AllBitsSet(unsigned char shadow_byte, uptr start_bit, uptr end_bit) {
  const unsigned char mask = BitMask(start_bit, end_bit);
  return (shadow_byte & mask) == mask;
}

// Sets (or clears) bits [`start_bit`, `end_bit`) of `*shadow_byte`.
void SetBits(unsigned char* shadow_byte, uptr start_bit, uptr end_bit,
             bool is_copy) {
  const unsigned char mask = BitMask(start_bit, end_bit);
  if (is_copy)
    *shadow_byte |= mask;
  else
    *shadow_byte &= static_cast<unsigned char>(~mask);
}

uptr BytesToShadowBits(uptr num_bytes) {
  // Each application byte maps to one shadow bit.
  return num_bytes;
}

}  // namespace

void InitializeShadowMemory() {
  g_copy_shadow = ShadowMemory::Create("copyprof");
}

void MarkApplicationMemory(const void* app_addr, uptr num_bytes, bool is_copy) {
  CHECK_GT(num_bytes, 0);
  const uptr addr = reinterpret_cast<uptr>(app_addr);
  unsigned char* shadow =
      reinterpret_cast<unsigned char*>(g_copy_shadow.MemToShadow(addr));

  // An application range need not start on a shadow byte boundary, so it is
  // updated in three steps: the leading (possibly partial) shadow byte, the
  // whole shadow bytes in the middle, and the trailing partial byte.
  uptr num_shadow_bits = BytesToShadowBits(num_bytes);
  if (uptr start_bit = addr % kBitsPerShadowByte; start_bit > 0) {
    uptr end_bit =
        start_bit + Min(kBitsPerShadowByte - start_bit, num_shadow_bits);
    SetBits(shadow++, start_bit, end_bit, is_copy);
    num_shadow_bits -= end_bit - start_bit;
  }
  if (uptr full_bytes = num_shadow_bits / kBitsPerShadowByte; full_bytes > 0) {
    internal_memset(shadow, is_copy ? 0xFF : 0, full_bytes);
    shadow += full_bytes;
    num_shadow_bits -= full_bytes * kBitsPerShadowByte;
  }
  if (num_shadow_bits > 0)
    SetBits(shadow, /*start_bit=*/0, num_shadow_bits, is_copy);
}

bool IsMarkedAsCopy(const void* app_addr, uptr num_bytes) {
  CHECK_GT(num_bytes, 0);
  const uptr addr = reinterpret_cast<uptr>(app_addr);
  const auto* shadow =
      reinterpret_cast<const unsigned char*>(g_copy_shadow.MemToShadow(addr));

  uptr num_shadow_bits = BytesToShadowBits(num_bytes);
  if (uptr start_bit = addr % kBitsPerShadowByte; start_bit > 0) {
    uptr end_bit =
        start_bit + Min(kBitsPerShadowByte - start_bit, num_shadow_bits);
    if (!AllBitsSet(*shadow++, start_bit, end_bit))
      return false;
    num_shadow_bits -= end_bit - start_bit;
  }
  uptr full_bytes = num_shadow_bits / kBitsPerShadowByte;
  for (uptr i = 0; i < full_bytes; ++i) {
    if (shadow[i] != 0xFF)
      return false;
  }
  shadow += full_bytes;
  num_shadow_bits -= full_bytes * kBitsPerShadowByte;
  return num_shadow_bits == 0 ||
         AllBitsSet(*shadow, /*start_bit=*/0, num_shadow_bits);
}

}  // namespace __copyprof
