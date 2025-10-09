// Check that memset() call from a shared library gets intercepted.
// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: %clangxx_asan -O0 %s -DSHARED_LIB \
// RUN:     -shared -o %dynamiclib -fPIC %ld_flags_rpath_so
// RUN: %clangxx_asan -O0 %s -o %t.dir/EXE %ld_flags_rpath_exe && \
// RUN:     not %run %t.dir/EXE 2>&1 | FileCheck %s

#include <stdio.h>
#include <string.h>

#if defined(SHARED_LIB)
extern "C"
void my_memset(void *p, size_t sz) {
  memset(p, 0, sz);
}
#else
extern "C" void my_memset(void *p, size_t sz);

int main(int argc, char *argv[]) {
  char buf[10];
  my_memset(buf, 11);
  // CHECK: {{.*ERROR: AddressSanitizer: stack-buffer-overflow}}
  // CHECK: {{WRITE of size 11 at 0x.* thread T0}}
  // CHECK: {{0x.* in my_memset .*interception-in-shared-lib-test.cpp:}}[[@LINE-10]]
  return 0;
}
#endif
