// RUN: %clangxx_msan -DGETC -O0 -g -xc++ %s -o %t && %run %t
// RUN: %clangxx_msan -DGETC -O3 -g -xc++ %s -o %t && %run %t
// RUN: %clang_msan -DGETC -O0 -g %s -o %t && %run %t
// RUN: %clang_msan -DGETC -O3 -g %s -o %t && %run %t

// RUN: %clangxx_msan -DGETCHAR -O0 -g -xc++ %s -o %t && %run %t
// RUN: %clangxx_msan -DGETCHAR -O3 -g -xc++ %s -o %t && %run %t
// RUN: %clang_msan -DGETCHAR -O0 -g %s -o %t && %run %t
// RUN: %clang_msan -DGETCHAR -O3 -g %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>
#include <unistd.h>

int main() {
  FILE *stream = fopen("/dev/zero", "r");
  flockfile (stream);
  int c;
#if defined(GETCHAR)
  int res = dup2(fileno(stream), 0);
  assert(res == 0);
  c = getchar_unlocked();
#elif defined(GETC)
  c = getc_unlocked (stream);
#endif
  funlockfile (stream);
  if (c == EOF)
    return 1;
  printf("%c\n", (char)c);
  fclose(stream);
  return 0;
}
