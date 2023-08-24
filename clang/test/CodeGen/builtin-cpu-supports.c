// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s| FileCheck %s

// Test that we have the structure definition, the gep offsets, the name of the
// global, the bit grab, and the icmp correct.
extern void a(const char *);

// CHECK: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }
// CHECK: @__cpu_features2 = external dso_local global [3 x i32]

int main(void) {
  __builtin_cpu_init();

  // CHECK: call void @__cpu_indicator_init

  if (__builtin_cpu_supports("sse4.2"))
    a("sse4.2");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, ptr @__cpu_model, i32 0, i32 3, i32 0)
  // CHECK: [[AND:%[^ ]+]] = and i32 [[LOAD]], 256
  // CHECK: = icmp eq i32 [[AND]], 256

  if (__builtin_cpu_supports("gfni"))
    a("gfni");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK: [[AND:%[^ ]+]] = and i32 [[LOAD]], 1
  // CHECK: = icmp eq i32 [[AND]], 1

  return 0;
}

// CHECK: declare dso_local void @__cpu_indicator_init()
