// Regression test for
// http://code.google.com/p/address-sanitizer/issues/detail?id=19
// Bug description:
// 1. application dlopens foo.so
// 2. asan registers all globals from foo.so
// 3. application dlcloses foo.so
// 4. application mmaps some memory to the location where foo.so was before
// 5. application starts using this mmaped memory, but asan still thinks there
// are globals.
// 6. BOOM

// This sublte test assumes that after a foo.so is dlclose-d
// we can mmap the region of memory that has been occupied by the library.
// It works on x86 Linux, but not necessary anywhere else.
// REQUIRES: x86-target-arch

// RUN: %clangxx_asan -O0 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O0 %s %libdl -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O1 %s %libdl -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O2 %s %libdl -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O3 %s %libdl -o %t && %run %t 2>&1 | FileCheck %s

#if !defined(SHARED_LIB)
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <string>

#if defined(__FreeBSD__)
// The MAP_NORESERVE define has been removed in FreeBSD 11.x, and even before
// that, it was never implemented. So just define it to zero.
#undef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

using std::string;

typedef int *(fun_t)();

int main(int argc, char *argv[]) {
  string path = string(argv[0]) + "-so.so";
  size_t PageSize = sysconf(_SC_PAGESIZE);
  printf("opening %s ... \n", path.c_str());
  void *lib = dlopen(path.c_str(), RTLD_NOW);
  if (!lib) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }
  fun_t *get = (fun_t*)dlsym(lib, "get_address_of_static_var");
  if (!get) {
    printf("failed dlsym\n");
    return 1;
  }
  int *addr = get();
  assert(((size_t)addr % 32) == 0);  // should be 32-byte aligned.
  printf("addr: %p\n", addr);
  addr[0] = 1;  // make sure we can write there.

  // Now dlclose the shared library.
  printf("attempting to dlclose\n");
  if (dlclose(lib)) {
    printf("failed to dlclose\n");
    return 1;
  }
  // Now, the page where 'addr' is unmapped. Map it.
  size_t page_beg = ((size_t)addr) & ~(PageSize - 1);
  void *res = mmap((void*)(page_beg), PageSize,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE, -1, 0);
  if (res == (char*)-1L) {
    printf("failed to mmap\n");
    return 1;
  }
  addr[1] = 2;  // BOOM (if the bug is not fixed).
  printf("PASS\n");
  // CHECK: PASS
  return 0;
}
#else  // SHARED_LIB
#include <stdio.h>

static int pad1;
static int static_var;
static int pad2;

extern "C"
int *get_address_of_static_var() {
  return &static_var;
}

__attribute__((constructor))
void at_dlopen() {
  printf("%s: I am being dlopened\n", __FILE__);
}
__attribute__((destructor))
void at_dlclose() {
  printf("%s: I am being dlclosed\n", __FILE__);
}
#endif  // SHARED_LIB
