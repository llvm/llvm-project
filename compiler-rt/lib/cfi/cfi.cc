//===-------- cfi.cc ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the runtime support for the cross-DSO CFI.
//
//===----------------------------------------------------------------------===//

// FIXME: Intercept dlopen/dlclose.
// FIXME: Support diagnostic mode.
// FIXME: Harden:
//  * mprotect shadow, use mremap for updates
//  * something else equally important

#include <assert.h>
#include <elf.h>
#include <link.h>
#include <string.h>

typedef ElfW(Phdr) Elf_Phdr;
typedef ElfW(Ehdr) Elf_Ehdr;

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "ubsan/ubsan_init.h"
#include "ubsan/ubsan_flags.h"

static uptr __cfi_shadow;
static constexpr uptr kShadowGranularity = 12;
static constexpr uptr kShadowAlign = 1UL << kShadowGranularity; // 4096

static constexpr uint16_t kInvalidShadow = 0;
static constexpr uint16_t kUncheckedShadow = 0xFFFFU;

static uint16_t *mem_to_shadow(uptr x) {
  return (uint16_t *)(__cfi_shadow + ((x >> kShadowGranularity) << 1));
}

typedef int (*CFICheckFn)(u64, void *, void *);

class ShadowValue {
  uptr addr;
  uint16_t v;
  explicit ShadowValue(uptr addr, uint16_t v) : addr(addr), v(v) {}

public:
  bool is_invalid() const { return v == kInvalidShadow; }

  bool is_unchecked() const { return v == kUncheckedShadow; }

  CFICheckFn get_cfi_check() const {
    assert(!is_invalid() && !is_unchecked());
    uptr aligned_addr = addr & ~(kShadowAlign - 1);
    uptr p = aligned_addr - (((uptr)v - 1) << kShadowGranularity);
    return reinterpret_cast<CFICheckFn>(p);
  }

  // Load a shadow valud for the given application memory address.
  static const ShadowValue load(uptr addr) {
    return ShadowValue(addr, *mem_to_shadow(addr));
  }
};

static void fill_shadow_constant(uptr begin, uptr end, uint16_t v) {
  assert(v == kInvalidShadow || v == kUncheckedShadow);
  uint16_t *shadow_begin = mem_to_shadow(begin);
  uint16_t *shadow_end = mem_to_shadow(end - 1) + 1;
  memset(shadow_begin, v, (shadow_end - shadow_begin) * sizeof(*shadow_begin));
}

static void fill_shadow(uptr begin, uptr end, uptr cfi_check) {
  assert((cfi_check & (kShadowAlign - 1)) == 0);

  // Don't fill anything below cfi_check. We can not represent those addresses
  // in the shadow, and must make sure at codegen to place all valid call
  // targets above cfi_check.
  uptr p = Max(begin, cfi_check);
  uint16_t *s = mem_to_shadow(p);
  uint16_t *s_end = mem_to_shadow(end - 1) + 1;
  uint16_t sv = ((p - cfi_check) >> kShadowGranularity) + 1;
  for (; s < s_end; s++, sv++)
    *s = sv;

  // Sanity checks.
  uptr q = p & ~(kShadowAlign - 1);
  for (; q < end; q += kShadowAlign) {
    assert((uptr)ShadowValue::load(q).get_cfi_check() == cfi_check);
    assert((uptr)ShadowValue::load(q + kShadowAlign / 2).get_cfi_check() ==
           cfi_check);
    assert((uptr)ShadowValue::load(q + kShadowAlign - 1).get_cfi_check() ==
           cfi_check);
  }
}

// This is a workaround for a glibc bug:
// https://sourceware.org/bugzilla/show_bug.cgi?id=15199
// Other platforms can, hopefully, just do
//    dlopen(RTLD_NOLOAD | RTLD_LAZY)
//    dlsym("__cfi_check").
static uptr find_cfi_check_in_dso(dl_phdr_info *info) {
  const ElfW(Dyn) *dynamic = nullptr;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
      dynamic =
          (const ElfW(Dyn) *)(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
      break;
    }
  }
  if (!dynamic) return 0;
  uptr strtab = 0, symtab = 0;
  for (const ElfW(Dyn) *p = dynamic; p->d_tag != PT_NULL; ++p) {
    if (p->d_tag == DT_SYMTAB)
      symtab = p->d_un.d_ptr;
    else if (p->d_tag == DT_STRTAB)
      strtab = p->d_un.d_ptr;
  }

  if (symtab > strtab) {
    VReport(1, "Can not handle: symtab > strtab (%p > %zx)\n", symtab, strtab);
    return 0;
  }

  // Verify that strtab and symtab are inside of the same LOAD segment.
  // This excludes VDSO, which has (very high) bogus strtab and symtab pointers.
  int phdr_idx;
  for (phdr_idx = 0; phdr_idx < info->dlpi_phnum; phdr_idx++) {
    const Elf_Phdr *phdr = &info->dlpi_phdr[phdr_idx];
    if (phdr->p_type == PT_LOAD) {
      uptr beg = info->dlpi_addr + phdr->p_vaddr;
      uptr end = beg + phdr->p_memsz;
      if (strtab >= beg && strtab < end && symtab >= beg && symtab < end)
        break;
    }
  }
  if (phdr_idx == info->dlpi_phnum) {
    // Nope, either different segments or just bogus pointers.
    // Can not handle this.
    VReport(1, "Can not handle: symtab %p, strtab %zx\n", symtab, strtab);
    return 0;
  }

  for (const ElfW(Sym) *p = (const ElfW(Sym) *)symtab; (ElfW(Addr))p < strtab;
       ++p) {
    char *name = (char*)(strtab + p->st_name);
    if (strcmp(name, "__cfi_check") == 0) {
      assert(p->st_info == ELF32_ST_INFO(STB_GLOBAL, STT_FUNC));
      uptr addr = info->dlpi_addr + p->st_value;
      return addr;
    }
  }
  return 0;
}

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *data) {
  uptr cfi_check = find_cfi_check_in_dso(info);
  if (cfi_check)
    VReport(1, "Module '%s' __cfi_check %zx\n", info->dlpi_name, cfi_check);

  for (int i = 0; i < info->dlpi_phnum; i++) {
    const Elf_Phdr *phdr = &info->dlpi_phdr[i];
    if (phdr->p_type == PT_LOAD) {
      // Jump tables are in the executable segment.
      // VTables are in the non-executable one.
      // Need to fill shadow for both.
      // FIXME: reject writable if vtables are in the r/o segment. Depend on
      // PT_RELRO?
      uptr cur_beg = info->dlpi_addr + phdr->p_vaddr;
      uptr cur_end = cur_beg + phdr->p_memsz;
      if (cfi_check) {
        VReport(1, "   %zx .. %zx\n", cur_beg, cur_end);
        fill_shadow(cur_beg, cur_end, cfi_check ? cfi_check : (uptr)(-1));
      } else {
        fill_shadow_constant(cur_beg, cur_end, kUncheckedShadow);
      }
    }
  }
  return 0;
}

// Fill shadow for the initial libraries.
static void init_shadow() {
  dl_iterate_phdr(dl_iterate_phdr_cb, nullptr);
}

static ALWAYS_INLINE void CfiSlowPathCommon(u64 CallSiteTypeId, void *Ptr,
                                            void *DiagData) {
  uptr Addr = (uptr)Ptr;
  VReport(3, "__cfi_slowpath: %llx, %p\n", CallSiteTypeId, Ptr);
  ShadowValue sv = ShadowValue::load(Addr);
  if (sv.is_invalid()) {
    // FIXME: call the ubsan handler if DiagData != nullptr?
    Report(
        "CFI: invalid memory region for a function pointer (shadow==0): %p\n",
        Ptr);
    if (DiagData)
      return;
    else
      Die();
  }
  if (sv.is_unchecked()) {
    VReport(2, "CFI: unchecked call (shadow=FFFF): %p\n", Ptr);
    return;
  }
  CFICheckFn cfi_check = sv.get_cfi_check();
  VReport(2, "__cfi_check at %p\n", cfi_check);
  cfi_check(CallSiteTypeId, Ptr, DiagData);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
__cfi_slowpath(u64 CallSiteTypeId, void *Ptr) {
  CfiSlowPathCommon(CallSiteTypeId, Ptr, nullptr);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
__cfi_slowpath_diag(u64 CallSiteTypeId, void *Ptr, void *DiagData) {
  CfiSlowPathCommon(CallSiteTypeId, Ptr, DiagData);
}

static void InitializeFlags() {
  SetCommonFlagsDefaults();
#ifdef CFI_ENABLE_DIAG
  __ubsan::Flags *uf = __ubsan::flags();
  uf->SetDefaults();
#endif

  FlagParser cfi_parser;
  RegisterCommonFlags(&cfi_parser);
  cfi_parser.ParseString(GetEnv("CFI_OPTIONS"));

#ifdef CFI_ENABLE_DIAG
  FlagParser ubsan_parser;
  __ubsan::RegisterUbsanFlags(&ubsan_parser, uf);
  RegisterCommonFlags(&ubsan_parser);

  const char *ubsan_default_options = __ubsan::MaybeCallUbsanDefaultOptions();
  ubsan_parser.ParseString(ubsan_default_options);
  ubsan_parser.ParseString(GetEnv("UBSAN_OPTIONS"));
#endif

  SetVerbosity(common_flags()->verbosity);

  if (Verbosity()) ReportUnrecognizedFlags();

  if (common_flags()->help) {
    cfi_parser.PrintFlagDescriptions();
  }
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
#if !SANITIZER_CAN_USE_PREINIT_ARRAY
// On ELF platforms, the constructor is invoked using .preinit_array (see below)
__attribute__((constructor(0)))
#endif
void __cfi_init() {
  SanitizerToolName = "CFI";
  InitializeFlags();

  uptr vma = GetMaxVirtualAddress();
  // Shadow is 2 -> 2**kShadowGranularity.
  uptr shadow_size = (vma >> (kShadowGranularity - 1)) + 1;
  VReport(1, "CFI: VMA size %zx, shadow size %zx\n", vma, shadow_size);
  void *shadow = MmapNoReserveOrDie(shadow_size, "CFI shadow");
  VReport(1, "CFI: shadow at %zx .. %zx\n", shadow,
          reinterpret_cast<uptr>(shadow) + shadow_size);
  __cfi_shadow = (uptr)shadow;
  init_shadow();

#ifdef CFI_ENABLE_DIAG
  __ubsan::InitAsPlugin();
#endif
}

#if SANITIZER_CAN_USE_PREINIT_ARRAY
// On ELF platforms, run cfi initialization before any other constructors.
// On other platforms we use the constructor attribute to arrange to run our
// initialization early.
extern "C" {
__attribute__((section(".preinit_array"),
               used)) void (*__cfi_preinit)(void) = __cfi_init;
}
#endif
