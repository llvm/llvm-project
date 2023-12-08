// Checks that the __sanitizer_on_print hook gets the exact same sanitizer
// report as what is printed to stderr.
//
// RUN: %clangxx %s -o %t
// RUN: %run %t %t-onprint.txt 2>%t-stderr.txt || true
// RUN: diff %t-onprint.txt %t-stderr.txt
//
// UNSUPPORTED: android

#include <cassert>
#include <cstdlib>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int f;
volatile void *buf;
volatile char sink;

__attribute__((disable_sanitizer_instrumentation)) extern "C" void
__sanitizer_on_print(const char *str) {
  write(f, str, strlen(str));
}

int main(int argc, char *argv[]) {
  assert(argc >= 2);
  f = open(argv[1], O_CREAT | O_TRUNC | O_WRONLY, 0666);

  // Use-after-free to trigger ASan/TSan reports.
  void *ptr = malloc(1);
  buf = ptr;
  free(ptr);
  sink = *static_cast<char *>(ptr);
  return 0;
}
