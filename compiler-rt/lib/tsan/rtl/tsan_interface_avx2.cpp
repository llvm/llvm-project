#include "tsan_interface_avx2.h"

#include <immintrin.h>
#include <inttypes.h>
#include <stdint.h>
#include <unistd.h>

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_ptrauth.h"
#include "tsan_interface_ann.h"
#include "tsan_rtl.h"

#define CALLERPC ((uptr)__builtin_return_address(0))

using namespace __tsan;

#ifdef __AVX__
extern "C" void __tsan_scatter_vector4(__m256i vaddr, int size, uint8_t mask) {
  uptr addr[4] = {};
  _mm256_store_si256((__m256i *)addr, vaddr);
  uptr pc = CALLERPC;
  ThreadState *thr = cur_thread();
  for (int i = 0; i < 4; i++)
    if ((mask >> i) & 1)
      UnalignedMemoryAccess(thr, pc, addr[i], size, kAccessWrite);
}

extern "C" void __tsan_gather_vector4(__m256i vaddr, int size, uint8_t mask) {
  uptr addr[4] = {};
  _mm256_store_si256((__m256i *)addr, vaddr);
  uptr pc = CALLERPC;
  ThreadState *thr = cur_thread();
  for (int i = 0; i < 4; i++)
    if ((mask >> i) & 1)
      UnalignedMemoryAccess(thr, pc, addr[i], size, kAccessRead);
}
#endif /*__AVX__*/
