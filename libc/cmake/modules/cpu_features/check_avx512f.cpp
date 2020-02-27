#if !defined __AVX512F__
#error "missing __AVX512F__"
#endif
#include <immintrin.h>
int main() {
  (void)_mm512_undefined();
  return 0;
}
