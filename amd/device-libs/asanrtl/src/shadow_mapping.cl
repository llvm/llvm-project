/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "shadow_mapping.h"

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
