// Test that custom memmove implementations instrumented by HWASan do not cause
// false positives.

// RUN: %clang_hwasan %s -o %t
// RUN: %run %t

#include <assert.h>
#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>

void *memmove(void *Dest, const void *Src, size_t N) {
  char *Tmp = (char *)malloc(N);
  char *D = (char *)Dest;
  const char *S = (const char *)Src;

  for (size_t I = 0; I < N; ++I)
    Tmp[I] = S[I];
  for (size_t I = 0; I < N; ++I)
    D[I] = Tmp[I];

  free(Tmp);
  return Dest;
}

int main() {
  __hwasan_enable_allocator_tagging();

  const size_t BufSize = 64;
  char *Buf = (char *)malloc(BufSize);

  for (size_t I = 0; I < BufSize; ++I)
    Buf[I] = I;
  memmove(Buf + BufSize / 2, Buf, BufSize / 2);
  for (size_t I = 0; I < BufSize; ++I)
    assert(Buf[I] == I % (BufSize / 2));

  free(Buf);
  return 0;
}
