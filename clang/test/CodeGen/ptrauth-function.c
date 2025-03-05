// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck -check-prefix=CHECK %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck -check-prefix=CHECK %s

void test_call();

// CHECK: define {{(dso_local )?}}void @test_direct_call()
void test_direct_call() {
  // CHECK: call void @test_call(){{$}}
  test_call();
}

// CHECK: define {{(dso_local )?}}void @test_indirect_call(ptr noundef %[[FP:.*]])
void test_indirect_call(void (*fp(void))) {
  // CHECK: %[[FP_ADDR:.*]] = alloca ptr, align 8
  // CHECK: store ptr %[[FP]], ptr %[[FP_ADDR]], align 8
  // CHECK: %[[V0:.*]] = load ptr, ptr %[[FP_ADDR]], align 8
  // CHECK: %[[CALL:.*]] = call ptr %[[V0]]() [ "ptrauth"(i32 0, i64 0) ]
  fp();
}

void abort();
// CHECK: define {{(dso_local )?}}void @test_direct_builtin_call()
void test_direct_builtin_call() {
  // CHECK: call void @abort() {{#[0-9]+$}}
  abort();
}

// CHECK-LABEL: define {{(dso_local )?}}void @test_memcpy_inline(
// CHECK-NOT: call{{.*}}memcpy

extern inline __attribute__((__always_inline__))
void *memcpy(void *d, const void *s, unsigned long) {
  return 0;
}

void test_memcpy_inline(char *d, char *s) {
  memcpy(d, s, 4);
}
