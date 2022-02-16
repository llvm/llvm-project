// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s| FileCheck %s

// Test that we have the structure definition, the gep offsets, the name of the
// global, the bit grab, and the icmp correct.
extern void a(const char *);

// CHECK: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

void intel(void) {
  if (__builtin_cpu_is("intel"))
    a("intel");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, i32* getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, { i32, i32, i32, [1 x i32] }* @__cpu_model, i32 0, i32 0)
  // CHECK: = icmp eq i32 [[LOAD]], 1
}

void amd(void) {
  if (__builtin_cpu_is("amd"))
    a("amd");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, i32* getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, { i32, i32, i32, [1 x i32] }* @__cpu_model, i32 0, i32 0)
  // CHECK: = icmp eq i32 [[LOAD]], 2
}

void atom(void) {
  if (__builtin_cpu_is("atom"))
    a("atom");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, i32* getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, { i32, i32, i32, [1 x i32] }* @__cpu_model, i32 0, i32 1)
  // CHECK: = icmp eq i32 [[LOAD]], 1
}

void amdfam10h(void) {
  if (__builtin_cpu_is("amdfam10h"))
    a("amdfam10h");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, i32* getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, { i32, i32, i32, [1 x i32] }* @__cpu_model, i32 0, i32 1)
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

void barcelona(void) {
  if (__builtin_cpu_is("barcelona"))
    a("barcelona");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, i32* getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, { i32, i32, i32, [1 x i32] }* @__cpu_model, i32 0, i32 2)
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

void nehalem(void) {
  if (__builtin_cpu_is("nehalem"))
    a("nehalem");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, i32* getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, { i32, i32, i32, [1 x i32] }* @__cpu_model, i32 0, i32 2)
  // CHECK: = icmp eq i32 [[LOAD]], 1
}
