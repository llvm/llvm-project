/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "asan_util.h"
#include "globals.h"
#include "shadow_mapping.h"

// fill shadow bytes of range [aligned_beg, aligned_beg+aligned_size)
// with value.
NO_SANITIZE_ADDR
static void
fill_shadowof(uptr aligned_beg, uptr aligned_size, s8 value) {
    u64 nbytes = aligned_size / SHADOW_GRANULARITY;
    __global s8 *shadow_beg = (__global s8*)MEM_TO_SHADOW(aligned_beg);
    for (; nbytes; nbytes--, shadow_beg++)
         *shadow_beg = value;
}

// poison the redzones around the global only if global is shadow granularity aligned.
NO_SANITIZE_ADDR
static void
poison_redzones(__global const struct device_global *g) {
    if (!is_aligned_by_granularity(g->beg))
      return;
    if (!is_aligned_by_granularity(g->size_with_redzone))
      return;

    uptr aligned_size = round_upto(g->size, SHADOW_GRANULARITY);
    uptr redzone_beg  = g->beg + aligned_size;
    uptr redzone_size = g->size_with_redzone - aligned_size;
    fill_shadowof(redzone_beg, redzone_size, kAsanGlobalRedzoneMagic);

    // poison partial redzones if any.
    // since SHADOW_GRANULARITY is 8 bytes we require only one shadow byte
    // to keep partially addressable bytes information.
    if (g->size != aligned_size) {
      uptr aligned_addr = g->beg + round_downto(g->size, SHADOW_GRANULARITY);
      __global s8 *shadow_addr = (__global s8*)MEM_TO_SHADOW(aligned_addr);
      *shadow_addr      = (s8) (g->size % SHADOW_GRANULARITY);
    }
}

// unpoison global and redzones around it only if global is shadow granularity aligned.
NO_SANITIZE_ADDR
static void
unpoison_redzones(__global const struct device_global *g) {
    if (!is_aligned_by_granularity(g->beg))
      return;
    if (!is_aligned_by_granularity(g->size_with_redzone))
      return;
    fill_shadowof(g->beg, g->size_with_redzone, 0);
}

// This function is called by one-workitem constructor kernel.
USED NO_INLINE NO_SANITIZE_ADDR
void
__asan_register_globals(uptr globals, uptr n) {
    __global struct device_global *dglobals = (__global struct device_global*) globals;
    for (uptr i = 0; i < n; i++)
       poison_redzones(&dglobals[i]);
}

// This function is called by one-workitem destructor kernel.
USED NO_INLINE NO_SANITIZE_ADDR
void
__asan_unregister_globals(uptr globals, uptr n) {
    __global struct device_global* dglobals = (__global struct device_global*) globals;
    for (uptr i = 0; i < n; i++)
       unpoison_redzones(&dglobals[i]);
}

USED NO_INLINE NO_SANITIZE_ADDR
void
__asan_register_elf_globals(uptr flag, uptr start, uptr stop)
{
    if (!start)
        return;

    __global uptr *f = (__global uptr *)flag;
    if (*f)
        return;

    __global struct device_global *b = (__global struct device_global *)start;
    __global struct device_global *e = (__global struct device_global *)stop;

    __asan_register_globals(start, e - b);

    *f = 1;
}

USED NO_INLINE NO_SANITIZE_ADDR
void
__asan_unregister_elf_globals(uptr flag, uptr start, uptr stop)
{
    if (!start)
        return;

    __global uptr *f = (__global uptr *)flag;
    if (!*f)
        return;

    __global struct device_global *b = (__global struct device_global *)start;
    __global struct device_global *e = (__global struct device_global *)stop;

    __asan_unregister_globals(start, e - b);

    *f = 0;
}

