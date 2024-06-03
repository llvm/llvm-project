// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck -check-prefix=CHECK %s

void test_call();

// CHECK-LABEL: define void @test_direct_call()
void test_direct_call() {
  // CHECK: call void @test_call(){{$}}
  test_call();
}

void abort();
// CHECK-LABEL: define void @test_direct_builtin_call()
void test_direct_builtin_call() {
  // CHECK: call void @abort() {{#[0-9]+$}}
  abort();
}

// CHECK-LABEL: define void @test_memcpy_inline(
// CHECK-NOT: call{{.*}}memcpy

extern inline __attribute__((__always_inline__))
void *memcpy(void *d, const void *s, unsigned long) {
  return 0;
}

void test_memcpy_inline(char *d, char *s) {
  memcpy(d, s, 4);
}
