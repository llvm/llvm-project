// REQUIRES: lld-available
// XFAIL: powerpc64-target-arch

// RUN: %clang_profgen -fuse-ld=lld -fcoverage-mapping -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile=%t.profdata 2>&1 | FileCheck %s --match-full-lines

// CHECK:   32|      0|#define MY_ID(x) ((x) ? 1 : 2)
// CHECK:   33|       |
// CHECK:   34|       |#define MY_LOG(fmt, ...)            \
// CHECK:   35|      1|  {                                 \
// CHECK:   36|      1|    if (enabled) {                  \
// CHECK:   37|      0|      printf(fmt, ## __VA_ARGS__);  \
// CHECK:   38|      0|    }                               \
// CHECK:   39|      1|  }
// CHECK:   40|       |
// CHECK:   41|      1|int main(int argc, char *argv[]) {
// CHECK:   42|      1|  enabled = argc > 2;
// CHECK:   43|      1|  MY_LOG("%d, %s, %d\n",
// CHECK:   44|      0|         MY_ID(argc > 3),
// CHECK:   45|      0|         "a",
// CHECK:   46|      0|         1);
// CHECK:   47|      1|  return 0;
// CHECK:   48|      1|}

#include <stdio.h>

static int enabled = 0;

// clang-format off
#define MY_ID(x) ((x) ? 1 : 2)

#define MY_LOG(fmt, ...)            \
  {                                 \
    if (enabled) {                  \
      printf(fmt, ## __VA_ARGS__);  \
    }                               \
  }

int main(int argc, char *argv[]) {
  enabled = argc > 2;
  MY_LOG("%d, %s, %d\n",
         MY_ID(argc > 3),
         "a",
         1);
  return 0;
}
// clang-format on
