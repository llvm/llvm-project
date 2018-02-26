// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s| FileCheck %s

// Test that we have the structure definition, the gep offsets, the name of the
// global, the bit grab, and the icmp correct.
extern void a(const char *);

int main() {
  __builtin_cpu_init();

  // CHECK: call void @__cpu_indicator_init

  if (__builtin_cpu_supports("sse4.2"))
    a("sse4.2");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, i32* getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, { i32, i32, i32, [1 x i32] }* @__cpu_model, i32 0, i32 3, i32 0)
  // CHECK: [[AND:%[^ ]+]] = and i32 [[LOAD]], 256
  // CHECK: = icmp ne i32 [[AND]], 0

  return 0;
}
