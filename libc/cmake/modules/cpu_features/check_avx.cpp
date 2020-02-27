#if !defined __AVX__
#error "missing __AVX__"
#endif
#include <immintrin.h>
int main() {
  (void)_mm256_set1_epi8('0');
  return 0;
}
