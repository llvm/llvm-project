// RUN: %clang_hwasan %s -o %t
// RUN: %run %t 0
// RUN: not %run %t 1 2>&1 | FileCheck --check-prefixes=CHECK,RSYM %s
// RUN: not %env_hwasan_opts=symbolize=0 %run %t 1 2>&1 | FileCheck --check-prefixes=CHECK,RNOSYM %s
// RUN: not %run %t -1 2>&1 | FileCheck --check-prefixes=CHECK,LSYM %s
// RUN: not %env_hwasan_opts=symbolize=0 %run %t -1 2>&1 | FileCheck --check-prefixes=CHECK,LNOSYM %s

// Test with and without optimizations, with and without PIC, since different
// backend passes run depending on these flags.
// RUN: %clang_hwasan -fno-pic %s -o %t
// RUN: not %run %t 1 2>&1 | FileCheck --check-prefixes=CHECK,RSYM %s
// RUN: %clang_hwasan -fno-pic -O2 %s -o %t
// RUN: not %run %t 1 2>&1 | FileCheck --check-prefixes=CHECK,RSYM %s
// RUN: %clang_hwasan -O2 %s -o %t
// RUN: not %run %t 1 2>&1 | FileCheck --check-prefixes=CHECK,RSYM %s

// RUN: %clang_hwasan -DUSE_NOSANITIZE %s -o %t && %run %t 0
// RUN: %clang_hwasan -DUSE_NOSANITIZE %s -o %t && %run %t 1
// RUN: %clang_hwasan -DUSE_NOSANITIZE %s -o %t -fno-pic && %run %t 1
// RUN: %clang_hwasan -DUSE_NOSANITIZE %s -o %t -O2 && %run %t 1
// RUN: %clang_hwasan -DUSE_NOSANITIZE %s -o %t -fno-pic -O2 && %run %t 1

// REQUIRES: pointer-tagging

#include <stdlib.h>

int a = 1;
#ifdef USE_NOSANITIZE
__attribute__((no_sanitize("hwaddress"))) int x = 1;
#else // USE_NOSANITIZE
int x = 1;
#endif // USE_NOSANITIZE
int b = 1;

int atoi(const char *);

int main(int argc, char **argv) {
  // CHECK: Cause: global-overflow
  // RSYM: is located 0 bytes to the right of 4-byte global variable x {{.*}} in {{.*}}global.c.tmp
  // RNOSYM: is located to the right of a 4-byte global variable in
  // RNOSYM-NEXT: #0 0x{{.*}} ({{.*}}global.c.tmp+{{.*}})
  // LSYM: is located 4 bytes to the left of 4-byte global variable x {{.*}} in {{.*}}global.c.tmp
  // LNOSYM: is located to the left of a 4-byte global variable in
  // LNOSYM-NEXT: #0 0x{{.*}} ({{.*}}global.c.tmp+{{.*}})
  // CHECK-NOT: can not describe
  (&x)[atoi(argv[1])] = 1;
}
