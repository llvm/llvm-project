#include <immintrin.h>

void test_compress_store(void *__P, __mmask8 __U, __m128d __A){
    return _mm_mask_compressstoreu_pd (__P, __U, __A);
}