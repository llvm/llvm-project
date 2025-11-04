/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "shadow_mapping.h"

static const __constant u8 kAsanHeapLeftRedzoneMagic = (u8)0xfa;
static const __constant u8 kAsanUserPoisonedMemoryMagic = (u8)0xf7;

NO_SANITIZE_ADDR
static uptr
range_check(uptr beg, uptr end) {
    uptr aligned_beg = round_downto(beg, SHADOW_GRANULARITY);
    uptr aligned_end = round_downto(end, SHADOW_GRANULARITY);
    uptr shadow_beg  = MEM_TO_SHADOW(aligned_beg);
    uptr shadow_end  = MEM_TO_SHADOW(aligned_end);
    uptr nbytes      = (shadow_end - shadow_beg)+1;
    uptr shadow_byte_count = 0;
    while (shadow_beg <= shadow_end) {
      s8 shadow_value = *(__global s8 *)shadow_beg;
      if (shadow_value)
        break;
      shadow_byte_count++;
      shadow_beg++;
    }
    if (shadow_byte_count == nbytes)
      return 0;
    uptr start_addr = round_downto(beg + (shadow_byte_count*SHADOW_GRANULARITY), SHADOW_GRANULARITY);
    return start_addr;
}

//check all application bytes in [beg,beg+size) range are accessible
USED NO_INLINE NO_SANITIZE_ADDR
uptr
__asan_region_is_poisoned(uptr beg, uptr size)
{
    uptr end  = beg + size - 1;
    uptr start_addr = range_check(beg, end);
    if (start_addr != 0) {
      // loop through the range to find accessible address.
      for (uptr addr = start_addr; addr <= end; ++addr) {
        if (is_address_poisoned(addr))
          return addr;
      }
    }
    return 0;
}

USED NO_INLINE NO_SANITIZE_ADDR
void
__asan_poison_region(ulong beg, ulong size)
{
    // Handle intial bytes if not aligned.
    if (!is_aligned_by_granularity(beg)) {
      ulong beg_round_downto = round_downto(beg, SHADOW_GRANULARITY);
      __global s8 *shadow_ptr = (__global s8 *)MEM_TO_SHADOW(beg_round_downto);
      s8 shadow_value = (s8) (beg - beg_round_downto);
      *shadow_ptr = shadow_value;
    }

    // Handle aligned bytes.
    ulong end  = round_downto(beg + size, SHADOW_GRANULARITY);
    ulong beg_round_upto = round_upto(beg, SHADOW_GRANULARITY);
    if (end > beg_round_upto) {
      u64 shadow_size = (end - beg_round_upto) / SHADOW_GRANULARITY;
      __global s8 *shadow_ptr = (__global s8 *)MEM_TO_SHADOW(beg_round_upto);
      __builtin_memset(shadow_ptr, kAsanHeapLeftRedzoneMagic, shadow_size);
    }
}

USED NO_SANITIZE_ADDR
void
__asan_poison_memory_region(const void *addr, uptr size)
{
    if (size == 0)
        return;

    uptr beg_addr = (uptr)addr;
    uptr end_addr = beg_addr + size;

    __global s8 *beg_sp = (__global s8 *)MEM_TO_SHADOW(beg_addr);
    s8 beg_off = (s8)(beg_addr & (SHADOW_GRANULARITY - 1));
    s8 beg_val = *beg_sp;

    __global s8 *end_sp = (__global s8 *)MEM_TO_SHADOW(end_addr);
    s8 end_off = (s8)(end_addr & (SHADOW_GRANULARITY - 1));
    s8 end_val = *end_sp;

    if (beg_sp == end_sp) {
        s8 val = beg_val;
        if (val > 0 && val <= end_off) {
            if (beg_off > 0)
                *beg_sp = (val < beg_off) ? val : beg_off;
            else
                *beg_sp = (s8)kAsanUserPoisonedMemoryMagic;
        }
        return;
    }

    if (beg_off > 0) {
        if (beg_val == 0)
            *beg_sp = beg_off;
        else
            *beg_sp = (beg_val < beg_off) ? beg_val : beg_off;
        beg_sp++;
    }

    __builtin_memset(beg_sp, kAsanUserPoisonedMemoryMagic, end_sp - beg_sp);

    if (end_val > 0 && end_val <= end_off)
        *end_sp = (s8)kAsanUserPoisonedMemoryMagic;
}

USED NO_SANITIZE_ADDR
void
__asan_unpoison_memory_region(const void *addr, uptr size)
{
    if (size == 0)
        return;

    uptr beg_addr = (uptr)addr;
    uptr end_addr = beg_addr + size;

    __global s8 *beg_sp = (__global s8 *)MEM_TO_SHADOW(beg_addr);
    s8 beg_off = (s8)(beg_addr & (SHADOW_GRANULARITY - 1));
    s8 beg_val = *beg_sp;

    __global s8 *end_sp = (__global s8 *)MEM_TO_SHADOW(end_addr);
    s8 end_off = (s8)(end_addr & (SHADOW_GRANULARITY - 1));
    s8 end_val = *end_sp;

    if (beg_sp == end_sp) {
        s8 val = beg_val;
        if (val != 0)
            *beg_sp = (val > end_off) ? val : end_off;
        return;
    }

    __builtin_memset(beg_sp, 0, end_sp - beg_sp);

    if (end_off > 0 && end_val != 0)
        *end_sp = (end_val > end_off) ? end_val : end_off;
}

USED NO_SANITIZE_ADDR
int
__asan_address_is_poisoned(const void *addr)
{
    return is_address_poisoned((uptr)addr);
}
