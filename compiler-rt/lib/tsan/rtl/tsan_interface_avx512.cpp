#include "tsan_interface_avx512.h"

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

#ifdef __AVX512F__
extern "C" void __tsan_scatter_vector8(__m512i vaddr, int size, uint8_t mask) {
  uptr addr[8] = {};
  __m256i v256_1 = _mm512_castsi512_si256(vaddr);
  __m256i v256_2 = _mm512_extracti64x4_epi64(vaddr, 1);
  _mm256_store_si256((__m256i *)addr, v256_1);
  _mm256_store_si256((__m256i *)&(addr[4]), v256_2);
  uptr pc = CALLERPC;
  ThreadState *thr = cur_thread();
  for (int i = 0; i < 8; i++)
    if ((mask >> i) & 1)
      UnalignedMemoryAccess(thr, pc, addr[i], size, kAccessWrite);
}

extern "C" void __tsan_gather_vector8(__m512i vaddr, int size, uint8_t mask) {
  uptr addr[8] = {};
  __m256i v256_1 = _mm512_castsi512_si256(vaddr);
  __m256i v256_2 = _mm512_extracti64x4_epi64(vaddr, 1);
  _mm256_store_si256((__m256i *)addr, v256_1);
  _mm256_store_si256((__m256i *)(&addr[4]), v256_2);
  uptr pc = CALLERPC;
  ThreadState *thr = cur_thread();
  for (int i = 0; i < 8; i++)
    if ((mask >> i) & 1)
      UnalignedMemoryAccess(thr, pc, addr[i], size, kAccessRead);
}
#endif /*__AVX512F__*/
