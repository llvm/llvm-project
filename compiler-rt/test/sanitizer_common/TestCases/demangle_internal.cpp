// RUN: %clangxx -O0 %s -o %t && %run %t

// REQUIRES: internal_symbolizer

#include <algorithm>
#include <assert.h>
#include <string.h>

extern "C" bool __sanitizer_symbolize_demangle(const char *Name, char *Buffer,
                                               int MaxLength);

int main() {
  char out[128];
  assert(!__sanitizer_symbolize_demangle("1A", out, sizeof(out)));

  const char name[] = "_Z3fooi";
  for (int i = 1; i < sizeof(out); ++i) {
    memset(out, 1, sizeof(out));
    assert(__sanitizer_symbolize_demangle(name, out, i) == (i > 8));
    assert(i < 9 || 0 == strncmp(out, "foo(int)", i - 1));
  }
}
