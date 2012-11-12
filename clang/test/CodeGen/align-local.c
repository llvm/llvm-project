// RUN: %clang_cc1 -emit-llvm < %s | grep "align 16" | count 2

typedef struct __attribute((aligned(16))) {int x[4];} ff;

int a() {
  ff a;
  struct {int x[4];} b __attribute((aligned(16)));
}
