#if defined(__linux__) || defined(__FreeBSD__) || defined(__Fuchsia__) ||      \
    (defined(__sun__) && defined(__svr4__)) || defined(__NetBSD__) ||          \
    defined(_AIX)

#include <elf.h>
#include <link.h>
#endif
#include <stdlib.h>
#include <string.h>

extern "C" {

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingTLS.h"
}

extern "C" {

#define PROF_TLS_CNTS_START INSTR_PROF_SECT_START(INSTR_PROF_TLS_CNTS_COMMON)
#define PROF_TLS_CNTS_STOP INSTR_PROF_SECT_STOP(INSTR_PROF_TLS_CNTS_COMMON)

extern char PROF_TLS_CNTS_START COMPILER_RT_VISIBILITY COMPILER_RT_WEAK;
extern char PROF_TLS_CNTS_STOP COMPILER_RT_VISIBILITY COMPILER_RT_WEAK;

COMPILER_RT_VISIBILITY char *__llvm_profile_begin_tls_counters(void) {
  return &PROF_TLS_CNTS_START;
}
COMPILER_RT_VISIBILITY char *__llvm_profile_end_tls_counters(void) {
  return &PROF_TLS_CNTS_STOP;
}

struct finalization_data {
  char *mod_begin;
  char *tls_img_begin;
  char *tls_img_end;
  char *cnts_begin;
  char *cnts_end;
};

// This is O(num_modules + num_counters) unfortunately.  If there were a
// mechanism to calculate the thread-local start of a thread-local section like
// there is a mechanism to calculate the static start of a static section (i.e.
// __start_$sectionname), that would simplify implementation a lot and make this
// just O(num_counters).
static int FindAndAddCounters_cb(struct dl_phdr_info *info, size_t size,
                                 void *data) {
  finalization_data *fdata = (finalization_data *)data;
  char *mod_begin = fdata->mod_begin;
  // We're looking for a match to the dladdr calculated based on PROF_CNTS_START
  if (mod_begin != (char *)info->dlpi_addr) {
    return 0;
  }

  if (info->dlpi_tls_data == NULL) {
    return 1;
  }

  const Elf64_Phdr *hdr = info->dlpi_phdr;
  const Elf64_Phdr *last_hdr = hdr + info->dlpi_phnum;

  const Elf64_Phdr *tls_hdr;
  for (; hdr != last_hdr; ++hdr) {
    if (hdr->p_type == PT_TLS) {
      tls_hdr = hdr;
      goto found_tls_ph;
    }
  }
  return 1;
found_tls_ph:
  uint64_t num_counters =
      __llvm_profile_get_num_counters(fdata->tls_img_begin, fdata->tls_img_end);
  uint64_t counter_size = __llvm_profile_counter_entry_size();

  // Calculate the offset of __llvm_prf_tls_cnts into the tls block for this
  // module. The addresses in use below correspond to the tls initialization
  // image, which is statically allocated for the module, rather than the TLS
  // block itself.
  uint64_t ph_true_vaddr =
      (uint64_t)info->dlpi_addr + (uint64_t)tls_hdr->p_vaddr;
  uint64_t tls_cnts_tlsblk_offset =
      (uint64_t)fdata->tls_img_begin - ph_true_vaddr;

  // Calculate the thread local copy of __llvm_prf_tls_cnts for this module.
  uint64_t tls_prf_cnts_modlocal_begin =
      (uint64_t)info->dlpi_tls_data + tls_cnts_tlsblk_offset;

  // We don't support single byte counters because they are also resilient to
  // thread synchronization issues and they are designed to avoid memory
  // overhead, which is the opposite of what TL counters do.
  // TODO: warn?
  if (counter_size == sizeof(uint64_t)) {
    uint64_t *tls_cnt = (uint64_t *)tls_prf_cnts_modlocal_begin;
    uint64_t *tls_end = (uint64_t *)tls_cnt + num_counters;
    uint64_t *cnt = (uint64_t *)fdata->cnts_begin;
    for (; tls_cnt != tls_end; tls_cnt++, cnt++) {
      __atomic_fetch_add(cnt, *tls_cnt, __ATOMIC_RELAXED);
    }
  }
  return 1;
}

COMPILER_RT_VISIBILITY
void __llvm_profile_tls_counters_finalize(void) {
  struct finalization_data fdata = {0};
  fdata.tls_img_begin = __llvm_profile_begin_tls_counters();
  fdata.tls_img_end = __llvm_profile_end_tls_counters();
  fdata.cnts_begin = __llvm_profile_begin_counters();
  fdata.cnts_end = __llvm_profile_end_counters();

  if (!fdata.tls_img_begin || !fdata.tls_img_end || !fdata.cnts_begin ||
      !fdata.cnts_end) {
    return;
  }

  Dl_info info;
  if (dladdr(fdata.cnts_begin, &info) == 0) {
    return;
  }
  fdata.mod_begin = (char *)info.dli_fbase;
  dl_iterate_phdr(FindAndAddCounters_cb, &fdata);
}
}
