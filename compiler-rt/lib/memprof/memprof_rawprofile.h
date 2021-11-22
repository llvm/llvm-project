#ifndef MEMPROF_RAWPROFILE_H_
#define MEMPROF_RAWPROFILE_H_

#include "memprof_mibmap.h"
#include "sanitizer_common/sanitizer_procmaps.h"

namespace __memprof {

// TODO: pull these in from MemProfData.inc
#define MEMPROF_RAW_MAGIC_64                                                   \
  (u64)255 << 56 | (u64)'m' << 48 | (u64)'p' << 40 | (u64)'r' << 32 |          \
      (u64)'o' << 24 | (u64)'f' << 16 | (u64)'r' << 8 | (u64)129

#define MEMPROF_RAW_VERSION 1ULL

u64 SerializeToRawProfile(MIBMapTy &BlockCache, MemoryMappingLayoutBase &Layout,
                          char *&Buffer);

} // namespace __memprof

#endif // MEMPROF_RAWPROFILE_H_
