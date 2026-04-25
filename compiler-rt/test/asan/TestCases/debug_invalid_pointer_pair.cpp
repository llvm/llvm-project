// Checks that the ASan debugging API for getting report information
// returns correct values for invalid pointer pairs.
// RUN: %clangxx_asan -O0 %s -o %t -mllvm -asan-detect-invalid-pointer-pair && %env_asan_opts=detect_invalid_pointer_pairs=1 not %run %t 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>

// If we use %p with MS CRTs, it comes out all upper case. Use %08x to get
// lowercase hex.
#ifdef _WIN32
#  ifdef _WIN64
#    define PTR_FMT "0x%08llx"
#  else
#    define PTR_FMT "0x%08x"
#  endif
// Solaris libc omits the leading 0x.
#elif defined(__sun__) && defined(__svr4__)
#  define PTR_FMT "0x%p"
#else
#  define PTR_FMT "%p"
#endif

char *p;
char *q;

int main() {
  // Disable stderr buffering. Needed on Windows.
  setvbuf(stderr, NULL, _IONBF, 0);

  p = (char *)malloc(42);
  q = (char *)malloc(42);

  fprintf(stderr, "p: " PTR_FMT "\n", p);
  // CHECK: p: 0x[[ADDR1:[0-9a-f]+]]
  fprintf(stderr, "q: " PTR_FMT "\n", q);
  // CHECK: q: 0x[[ADDR2:[0-9a-f]+]]

  // Trigger invalid pointer pair
  int res = p > q; // BOOM

  free(p);
  free(q);
  return res;
}

// Required for dyld macOS 12.0+
#if (__APPLE__)
__attribute__((weak))
#endif
extern "C" void __asan_on_error() {
  int present = __asan_report_present();
  fprintf(stderr, "%s\n", (present == 1) ? "report" : "");
  // CHECK: report

  const void *addr_first = NULL;
  size_t size_first = 0xbad;
  int is_first = __asan_get_report_first_address(&addr_first, &size_first);
  fprintf(stderr, "is_first: %d, addr_first: " PTR_FMT ", size_first: %zu\n",
          is_first, addr_first, size_first);
  // CHECK: is_first: 1, addr_first: 0x[[ADDR1]], size_first: 0

  const void *addr_second = NULL;
  size_t size_second = 0xbad;
  int is_second = __asan_get_report_second_address(&addr_second, &size_second);
  fprintf(stderr, "is_second: %d, addr_second: " PTR_FMT ", size_second: %zu\n",
          is_second, addr_second, size_second);
  // CHECK: is_second: 1, addr_second: 0x[[ADDR2]], size_second: 0
}

// CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
