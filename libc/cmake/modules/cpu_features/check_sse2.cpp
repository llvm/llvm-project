#if !defined __SSE2__
#error "missing __SSE2__"
#endif
#include <immintrin.h>
int main() {
  (void)_mm_set1_epi8('0');
  return 0;
}
