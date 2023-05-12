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

#include <inttypes.h>
#include <stdlib.h>

struct data {
  uint64_t x;
  uint64_t y;
};

// GlobalOpt may replace the current GV with a new boolean-typed GV. Previously,
// this resulted in the "nosanitize" getting dropped because while the data/code
// references to the GV were updated, the old metadata references weren't.
struct data *f() {
#ifdef USE_NOSANITIZE
  __attribute__((no_sanitize("hwaddress"))) static struct data x = {1, 0};
#else // USE_NOSANITIZE
  static struct data x = {1, 0};
#endif // USE_NOSANITIZE
  if (x.x == 1)
    x.x = 0;
  return &x;
}

int main(int argc, char **argv) {
  // CHECK: Cause: global-overflow
  // RSYM: is located 0 bytes after a 16-byte global variable f.x {{.*}} in {{.*}}global-with-reduction.c.tmp
  // RNOSYM: is located after a 16-byte global variable in
  // RNOSYM-NEXT: #0 0x{{.*}} ({{.*}}global-with-reduction.c.tmp+{{.*}})
  // LSYM: is located 16 bytes before a 16-byte global variable f.x {{.*}} in {{.*}}global-with-reduction.c.tmp
  // LNOSYM: is located before a 16-byte global variable in
  // LNOSYM-NEXT: #0 0x{{.*}} ({{.*}}global-with-reduction.c.tmp+{{.*}})
  // CHECK-NOT: can not describe
  f()[atoi(argv[1])].x = 1;
}
