// Checks that the ASan debugging API for getting report information reports
// memory overlap error details.
// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <string.h>

char buffer[10] = "hello";

int main() {
  // Disable stderr buffering. Needed on Windows.
  setvbuf(stderr, NULL, _IONBF, 0);

  // Trigger memcpy-param-overlap
  memcpy(buffer, buffer + 1, 3); // BOOM
  return 0;
}

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

// Required for dyld macOS 12.0+
#if (__APPLE__)
__attribute__((weak))
#endif
extern "C" void __asan_on_error() {
  int present = __asan_report_present();
  fprintf(stderr, "%s\n", (present == 1) ? "report" : "");
  // CHECK: report

  const void *addr_src = NULL;
  size_t size_src = 0;
  int is_src = __asan_get_report_src_address(&addr_src, &size_src);
  fprintf(stderr, "is_src: %d, addr_src: " PTR_FMT ", size_src: %zu\n", is_src,
          addr_src, size_src);
  // CHECK: is_src: 1, addr_src: 0x{{[0-9a-f]+}}, size_src: 3

  const void *addr_dest = NULL;
  size_t size_dest = 0;
  int is_dest = __asan_get_report_dest_address(&addr_dest, &size_dest);
  fprintf(stderr, "is_dest: %d, addr_dest: " PTR_FMT ", size_dest: %zu\n",
          is_dest, addr_dest, size_dest);
  // CHECK: is_dest: 1, addr_dest: 0x{{[0-9a-f]+}}, size_dest: 3
}

// CHECK: memcpy-param-overlap
