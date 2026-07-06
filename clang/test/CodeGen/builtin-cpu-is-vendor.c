// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s | FileCheck %s

// Test that __builtin_cpu_is emits the correct ABI value and field offset for
// every vendor (field offset 0) in llvm/include/llvm/TargetParser/
// X86TargetParser.def.
extern void a(const char *);

// CHECK: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

#define TEST_CPU_IS(NAME, STR)                                                 \
  void test_##NAME(void) {                                                     \
    if (__builtin_cpu_is(STR))                                                 \
      a(STR);                                                                  \
  }

// CHECK-LABEL: define{{.*}} void @test_intel(
// CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
// CHECK: = icmp eq i32 [[LOAD]], 1
TEST_CPU_IS(intel, "intel")

// CHECK: = icmp eq i32 {{.*}}, 2
TEST_CPU_IS(amd, "amd")

// CHECK: = icmp eq i32 {{.*}}, 4
TEST_CPU_IS(other, "other")
