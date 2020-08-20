// RUN: %clang_dfsan -fno-sanitize=dataflow -DCALLOC -c %s -o %t-calloc.o
// RUN: %clang_dfsan %s %t-calloc.o -o %t
// RUN: %run %t
//
// Tests that calling mmap() during during dfsan initialization works.

#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef CALLOC

// dfsan_init() installs interceptors via dlysm(), which calls calloc().
// Calling mmap() from here should work even if interceptors haven't been fully
// set up yet.
void *calloc(size_t Num, size_t Size) {
  size_t PageSize = getpagesize();
  Size = Size * Num;
  Size = (Size + PageSize - 1) & ~(PageSize - 1); // Round up to PageSize.
  void *Ret = mmap(NULL, Size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(Ret != MAP_FAILED);
  return Ret;
}

#else

int main() { return 0; }

#endif // CALLOC
