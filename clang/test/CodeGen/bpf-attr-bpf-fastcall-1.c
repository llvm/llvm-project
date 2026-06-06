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
// CHECK: define {{.*}} void @foo()
// CHECK: entry:
// CHECK:   call void @test() #[[call_attr:[0-9]+]]
// CHECK:   %[[ptr:.*]] = load ptr, ptr @ptr, align 8
// CHECK:   call void %[[ptr]]() #[[call_attr]]
// CHECK:   ret void

// CHECK: declare void @test() #[[func_attr:[0-9]+]]
// CHECK: attributes #[[func_attr]] = { {{.*}}"bpf_fastcall"{{.*}} }
// CHECK: attributes #[[call_attr]] = { "bpf_fastcall" }
