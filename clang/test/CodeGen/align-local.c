// RUN: %clang_cc1 -emit-llvm < %s | FileCheck %s

typedef struct __attribute((aligned(16))) {int x[4];} ff;

// CHECK: alloca %struct.ff, align 16
// CHECK: alloca %struct.anon, align 16
void a(void) {
  ff a;
  struct {int x[4];} b __attribute((aligned(16)));
}
