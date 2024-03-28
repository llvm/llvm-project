// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>

// If we use %p with MSVC, it comes out all upper case. Use %08x to get
// lowercase hex.
#ifdef _MSC_VER
# ifdef _WIN64
#  define PTR_FMT "0x%08llx"
# else
#  define PTR_FMT "0x%08x"
# endif
// Solaris libc omits the leading 0x.
#elif defined(__sun__) && defined(__svr4__)
# define PTR_FMT "0x%p"
#else
# define PTR_FMT "%p"
#endif

char *heap_ptr;

int main() {
  // Disable stderr buffering. Needed on Windows.
  setvbuf(stderr, NULL, _IONBF, 0);

  heap_ptr = (char *)malloc(10);
  fprintf(stderr, "heap_ptr: " PTR_FMT "\n", heap_ptr);
  // CHECK: heap_ptr: 0x[[ADDR:[0-9a-f]+]]

  free(heap_ptr);
  free(heap_ptr);  // BOOM
  return 0;
}

// Required for dyld macOS 12.0+
#if (__APPLE__)
__attribute__((weak))
#endif
extern "C" void
__asan_on_error() {
  int present = __asan_report_present();
  void *addr = __asan_get_report_address();
  const char *description = __asan_get_report_description();

  fprintf(stderr, "%s\n", (present == 1) ? "report present" : "");
  // CHECK: report present
  fprintf(stderr, "addr: " PTR_FMT "\n", addr);
  // CHECK: addr: {{0x0*}}[[ADDR]]
  fprintf(stderr, "description: %s\n", description);
  // CHECK: description: double-free
}

// CHECK: AddressSanitizer: attempting double-free on {{0x0*}}[[ADDR]] in thread T0
