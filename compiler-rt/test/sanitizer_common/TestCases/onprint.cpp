// Checks that the __sanitizer_on_print hook gets the exact same sanitizer
// report as what is printed to stderr.
//
// RUN: %clangxx %s -o %t
// RUN: %run %t %t-onprint.txt 2>%t-stderr.txt || true
// RUN: diff %t-onprint.txt %t-stderr.txt
//
// UNSUPPORTED: android

#include <cassert>
#include <cstdio>
#include <cstdlib>

FILE *f;
volatile void *buf;
volatile char sink;

extern "C" void __sanitizer_on_print(const char *str) {
  fprintf(f, "%s", str);
  fflush(f);
}

int main(int argc, char *argv[]) {
  assert(argc >= 2);
  f = fopen(argv[1], "w");

  // Use-after-free to trigger ASan/TSan reports.
  void *ptr = malloc(1);
  buf = ptr;
  free(ptr);
  sink = *static_cast<char *>(ptr);
  return 0;
}
