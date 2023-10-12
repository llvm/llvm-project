// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s | FileCheck %s \
// RUN:   --check-prefix=CHECK-X86
// RUN: %clang_cc1 -triple ppc64le-linux-gnu -emit-llvm < %s | FileCheck %s \
// RUN:   --check-prefix=CHECK-PPC

#ifndef __PPC__

// Test that we have the structure definition, the gep offsets, the name of the
// global, the bit grab, and the icmp correct.
extern void a(const char *);

// CHECK-X86: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }
// CHECK-X86: @__cpu_features2 = external dso_local global [3 x i32]

int main(void) {
  __builtin_cpu_init();

  // CHECK: call void @__cpu_indicator_init

  if (__builtin_cpu_supports("sse4.2"))
    a("sse4.2");

  // CHECK-X86: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, ptr @__cpu_model, i32 0, i32 3, i32 0)
  // CHECK-X86: [[AND:%[^ ]+]] = and i32 [[LOAD]], 256
  // CHECK-X86: = icmp eq i32 [[AND]], 256

  if (__builtin_cpu_supports("gfni"))
    a("gfni");

  // CHECK-X86: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_features2
  // CHECK-X86: [[AND:%[^ ]+]] = and i32 [[LOAD]], 1
  // CHECK-X86: = icmp eq i32 [[AND]], 1

  return 0;
}

// CHECK-X86: declare dso_local void @__cpu_indicator_init()

// CHECK-X86-LABEL: define{{.*}} @baseline(
// CHECK-X86:         [[LOAD:%.*]] = load i32, ptr getelementptr inbounds ([[[#]] x i32], ptr @__cpu_features2, i32 0, i32 1)
// CHECK-X86-NEXT:    and i32 [[LOAD]], -2147483648
int baseline() { return __builtin_cpu_supports("x86-64"); }

// CHECK-X86-LABEL: define{{.*}} @v2(
// CHECK-X86:         [[LOAD:%.*]] = load i32, ptr getelementptr inbounds ([[[#]] x i32], ptr @__cpu_features2, i32 0, i32 2)
// CHECK-X86-NEXT:    and i32 [[LOAD]], 1
int v2() { return __builtin_cpu_supports("x86-64-v2"); }

// CHECK-X86-LABEL: define{{.*}} @v3(
// CHECK-X86:         [[LOAD:%.*]] = load i32, ptr getelementptr inbounds ([[[#]] x i32], ptr @__cpu_features2, i32 0, i32 2)
// CHECK-X86-NEXT:    and i32 [[LOAD]], 2
int v3() { return __builtin_cpu_supports("x86-64-v3"); }

// CHECK-X86-LABEL: define{{.*}} @v4(
// CHECK-X86:         [[LOAD:%.*]] = load i32, ptr getelementptr inbounds ([[[#]] x i32], ptr @__cpu_features2, i32 0, i32 2)
// CHECK-X86-NEXT:    and i32 [[LOAD]], 4
int v4() { return __builtin_cpu_supports("x86-64-v4"); }
#else
int test(int a) {
// CHECK-PPC: [[CPUSUP:%[^ ]+]] = call i32 @llvm.ppc.fixed.addr.ld(i32 2)
// CHECK-PPC: [[AND:%[^ ]+]] = and i32 [[CPUSUP]], 8388608
// CHECK-PPC: icmp ne i32 [[AND]], 0
// CHECK-PPC: [[CPUSUP2:%[^ ]+]] = call i32 @llvm.ppc.fixed.addr.ld(i32 1)
// CHECK-PPC: [[AND2:%[^ ]+]] = and i32 [[CPUSUP2]], 67108864
// CHECK-PPC: icmp ne i32 [[AND2]], 0
// CHECK-PPC: [[CPUID:%[^ ]+]] = call i32 @llvm.ppc.fixed.addr.ld(i32 3)
// CHECK-PPC: icmp eq i32 [[CPUID]], 39
  if (__builtin_cpu_supports("arch_3_00")) // HWCAP2
    return a;
  else if (__builtin_cpu_supports("mmu"))  // HWCAP
    return a - 5;
  else if (__builtin_cpu_is("power7"))     // CPUID
    return a + a;
  return a + 5;
}
#endif
