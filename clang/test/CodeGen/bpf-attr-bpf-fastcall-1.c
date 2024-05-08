// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

#define __bpf_fastcall __attribute__((bpf_fastcall))

void test(void) __bpf_fastcall;
void (*ptr)(void) __bpf_fastcall;

void foo(void) {
  test();
  (*ptr)();
}

// CHECK: @ptr = global ptr null
// CHECK: define {{.*}} @foo()
// CHECK: entry:
// CHECK:   call void @test() #[[test_attr:[0-9]+]]
// CHECK:   %[[ptr:.*]] = load ptr, ptr @ptr, align 8
// CHECK:   call void %[[ptr]]() #[[test_attr]]
// CHECK:   ret void

// CHECK: declare void @test() #[[ptr_attr:[0-9]+]]
// CHECK: attributes #1 = { {{.*}}"bpf_fastcall"{{.*}} }
// CHECK: attributes #[[test_attr]] = { {{.*}}"bpf_fastcall"{{.*}} }
