#if !defined __SSE__
#error "missing __SSE__"
#endif
#include <immintrin.h>
int main() {
  (void)_mm_set_ss(1.0f);
  return 0;
}
