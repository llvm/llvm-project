//===-- sanitizer_procmaps_freebsd.cc -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Information about the process mappings (FreeBSD-specific parts).
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_FREEBSD
#include "sanitizer_common.h"
#include "sanitizer_freebsd.h"
#include "sanitizer_procmaps.h"

#include <unistd.h>
#include <sys/sysctl.h>
#include <sys/user.h>

// Fix 'kinfo_vmentry' definition on FreeBSD prior v9.2 in 32-bit mode.
#if SANITIZER_FREEBSD && (SANITIZER_WORDSIZE == 32)
# include <osreldate.h>
# if __FreeBSD_version <= 902001  // v9.2
#  define kinfo_vmentry xkinfo_vmentry
# endif
#endif

namespace __sanitizer {

void ReadProcMaps(ProcSelfMapsBuff *proc_maps) {
  const int Mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_VMMAP, getpid() };
  size_t Size = 0;
  int Err = sysctl(Mib, 4, NULL, &Size, NULL, 0);
  CHECK_EQ(Err, 0);
  CHECK_GT(Size, 0);

  size_t MmapedSize = Size * 4 / 3;
  void *VmMap = MmapOrDie(MmapedSize, "ReadProcMaps()");
  Size = MmapedSize;
  Err = sysctl(Mib, 4, VmMap, &Size, NULL, 0);
  CHECK_EQ(Err, 0);

  proc_maps->data = (char*)VmMap;
  proc_maps->mmaped_size = MmapedSize;
  proc_maps->len = Size;
}

bool MemoryMappingLayout::Next(uptr *start, uptr *end, uptr *offset,
                               char filename[], uptr filename_size,
                               uptr *protection) {
  char *last = proc_self_maps_.data + proc_self_maps_.len;
  if (current_ >= last) return false;
  uptr dummy;
  if (!start) start = &dummy;
  if (!end) end = &dummy;
  if (!offset) offset = &dummy;
  if (!protection) protection = &dummy;
  struct kinfo_vmentry *VmEntry = (struct kinfo_vmentry*)current_;

  *start = (uptr)VmEntry->kve_start;
  *end = (uptr)VmEntry->kve_end;
  *offset = (uptr)VmEntry->kve_offset;

  *protection = 0;
  if ((VmEntry->kve_protection & KVME_PROT_READ) != 0)
    *protection |= kProtectionRead;
  if ((VmEntry->kve_protection & KVME_PROT_WRITE) != 0)
    *protection |= kProtectionWrite;
  if ((VmEntry->kve_protection & KVME_PROT_EXEC) != 0)
    *protection |= kProtectionExecute;

  if (filename != NULL && filename_size > 0) {
    internal_snprintf(filename,
                      Min(filename_size, (uptr)PATH_MAX),
                      "%s", VmEntry->kve_path);
  }

  current_ += VmEntry->kve_structsize;

  return true;
}

}  // namespace __sanitizer

#endif  // SANITIZER_FREEBSD
