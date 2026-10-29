//===-- asan_shadow_setup.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Set up the shadow memory.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

// asan_fuchsia.cpp has their own InitializeShadowMemory implementation.
#if !SANITIZER_FUCHSIA

#  include "asan_internal.h"
#  include "asan_mapping.h"

namespace __asan {

static void ProtectGap(uptr addr, uptr size) {
  if (!flags()->protect_shadow_gap) {
    // The shadow gap is unprotected, so there is a chance that someone
    // is actually using this memory. Which means it needs a shadow...
    uptr GapShadowBeg = RoundDownTo(MEM_TO_SHADOW(addr), GetPageSizeCached());
    uptr GapShadowEnd =
        RoundUpTo(MEM_TO_SHADOW(addr + size), GetPageSizeCached()) - 1;
    if (Verbosity())
      Printf(
          "protect_shadow_gap=0:"
          " not protecting shadow gap, allocating gap's shadow\n"
          "|| `[%p, %p]` || ShadowGap's shadow ||\n",
          (void*)GapShadowBeg, (void*)GapShadowEnd);
    ReserveShadowMemoryRange(GapShadowBeg, GapShadowEnd,
                             "unprotected gap shadow");
    return;
  }
  VReport(2, "ProtectGap %p sz=%p\n", (void*)addr, (void*)size);
  __sanitizer::ProtectGap(addr, size, kZeroBaseShadowStart,
                          kZeroBaseMaxShadowStart);
}

static void MaybeReportLinuxPIEBug() {
#if SANITIZER_LINUX && \
    (defined(__x86_64__) || defined(__aarch64__) || SANITIZER_RISCV64)
  Report("This might be related to ELF_ET_DYN_BASE change in Linux 4.12.\n");
  Report(
      "See https://github.com/google/sanitizers/issues/856 for possible "
      "workarounds.\n");
#endif
}

void InitializeShadowMemory() {
  // Set the shadow memory address to uninitialized.
  __asan_shadow_memory_dynamic_address = kDefaultShadowSentinel;

  uptr shadow_start = kLowShadowBeg;
  // Detect if a dynamic shadow address must used and find a available location
  // when necessary. When dynamic address is used, the macro |kLowShadowBeg|
  // expands to |__asan_shadow_memory_dynamic_address| which is
  // |kDefaultShadowSentinel|.
  bool full_shadow_is_available = false;
  if (shadow_start == kDefaultShadowSentinel) {
    shadow_start = FindDynamicShadowStart();
    if (SANITIZER_LINUX) full_shadow_is_available = true;
  }
  // Update the shadow memory address (potentially) used by instrumentation.
  __asan_shadow_memory_dynamic_address = shadow_start;

  if (kLowShadowBeg) shadow_start -= GetMmapGranularity();

  if (!full_shadow_is_available)
    full_shadow_is_available =
        MemoryRangeIsAvailable(shadow_start, kHighShadowEnd);

#if SANITIZER_LINUX && defined(__x86_64__) && defined(_LP64) && \
    !ASAN_FIXED_MAPPING
  if (!full_shadow_is_available) {
    kMidMemBeg = kLowMemEnd < 0x3000000000ULL ? 0x3000000000ULL : 0;
    kMidMemEnd = kLowMemEnd < 0x3000000000ULL ? 0x4fffffffffULL : 0;
  }
#endif

  if (Verbosity()) PrintAddressSpaceLayout();

  if (full_shadow_is_available && kGaplessShadow) {
    // Normally, the shadow memory overlaps with the memory mappable
    // by the application, so we split shadow into "low" and "high"
    // with a protected gap in the middle (the shadow of the shadow).
    //
    // However, on some platforms, we can map the shadow above
    // the space normally addressable by the application. On these
    // platforms, we do not need a gap.

    // In the "gapless" configuration, there is only one shadow mapping
    // which covers all app memory i.e. from kLowMemBeg to kHighMemEnd.
    ReserveShadowMemoryRange(shadow_start, kHighShadowEnd, "shadow");

    // kLowShadowEnd, kHighShadowBeg are defined assuming there is a gap,
    // and this affects calls such as AddrIsInLowMem and AddrIsInHighMem.
    //
    // We want all of application memory to be in the "low mem" region and all
    // of the shadow to be in the "low shadow" region. However, kLowMemEnd
    // is defined differently in terms of the shadow base, which is always above
    // the actual app mem max (i.e. >4TB, kHighMemEnd). This means
    // (kLowMemBeg, kLowMemEnd) is a slight over-approximation of the low app
    // memory. However, it's still good enough for us because it includes
    // all app memory and no shadow memory, which we assert here.
    CHECK_GE(kLowMemEnd, kHighMemEnd);
    CHECK_LT(kLowMemEnd, kLowShadowBeg);
    CHECK_GE(kLowShadowEnd, kHighShadowEnd);

    // We don't use the "high mem" region, so we expect beg > end, to ensure
    // that AddrIsInHighMem/AddrIsInHighShadow always fails.
    CHECK_GT(kHighMemBeg, kHighMemEnd);
    CHECK_GT(kHighShadowBeg, kHighShadowEnd);

    // The shadow of the shadow may still technically be mappable by the
    // sanitizers or other tools, so we protect it here just to be safe.
    ProtectGap(
        MEM_TO_SHADOW(kLowShadowBeg),
        MEM_TO_SHADOW(kHighShadowEnd) - MEM_TO_SHADOW(kLowShadowBeg) + 1);
  } else if (full_shadow_is_available) {
    // mmap the low shadow plus at least one page at the left.
    if (kLowShadowBeg)
      ReserveShadowMemoryRange(shadow_start, kLowShadowEnd, "low shadow");
    // mmap the high shadow and protect the gap.
    // On targets where the shadow offset sits above all addressable memory
    // (e.g. Alpha's 42-bit user VAS with offset 0x70000000000), the shadow of
    // the highest address exceeds the highest address itself, so there is no
    // high memory region.  Skip both the high-shadow reservation and the gap
    // protect.
    if (MEM_TO_SHADOW(GetMaxUserVirtualAddress()) <
        GetMaxUserVirtualAddress()) {
      DCHECK_LE(kHighMemBeg, kHighMemEnd);
      ReserveShadowMemoryRange(kHighShadowBeg, kHighShadowEnd, "high shadow");
      ProtectGap(kShadowGapBeg, kShadowGapEnd - kShadowGapBeg + 1);
      CHECK_EQ(kShadowGapEnd, kHighShadowBeg - 1);
    }
  } else if (kMidMemBeg &&
             MemoryRangeIsAvailable(shadow_start, kMidMemBeg - 1) &&
             MemoryRangeIsAvailable(kMidMemEnd + 1, kHighShadowEnd)) {
    CHECK(kLowShadowBeg != kLowShadowEnd);
    // mmap the low shadow plus at least one page at the left.
    ReserveShadowMemoryRange(shadow_start, kLowShadowEnd, "low shadow");
    // mmap the mid shadow.
    ReserveShadowMemoryRange(kMidShadowBeg, kMidShadowEnd, "mid shadow");
    // mmap the high shadow.
    ReserveShadowMemoryRange(kHighShadowBeg, kHighShadowEnd, "high shadow");
    // protect the gaps.
    ProtectGap(kShadowGapBeg, kShadowGapEnd - kShadowGapBeg + 1);
    ProtectGap(kShadowGap2Beg, kShadowGap2End - kShadowGap2Beg + 1);
    ProtectGap(kShadowGap3Beg, kShadowGap3End - kShadowGap3Beg + 1);
  } else {
    // ASan's mappings can usually shadow the entire address space, even with
    // maximum ASLR entropy. However:
    // - On 32-bit systems, the maximum ASLR entropy (currently up to 16-bits
    //   == 256MB) is a significant chunk of the address space; reclaiming it
    //   by disabling ASLR might allow chonky binaries to run.
    // - On 64-bit systems, some settings (e.g., for Linux, unlimited stack
    //   size plus 31+ bits of entropy) can lead to an incompatible layout.
    TryReExecWithoutASLR();

    Report(
        "Shadow memory range interleaves with an existing memory mapping. "
        "ASan cannot proceed correctly. ABORTING.\n");
    Report("ASan shadow was supposed to be located in the [%p-%p] range.\n",
           (void*)shadow_start, (void*)kHighShadowEnd);
    MaybeReportLinuxPIEBug();
    DumpProcessMap();
    Die();
  }
}

}  // namespace __asan

#endif  // !SANITIZER_FUCHSIA
