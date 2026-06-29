// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -target-cpu pwr10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -target-cpu pwr10 -emit-llvm %s -o - | FileCheck %s

// Test MMA feature which requires pwr10
int __attribute__((target_clones("mma", "default")))
foo_mma(void) { return 0; }
// CHECK: define internal {{.*}}i32 @foo_mma.mma()
// CHECK: define internal {{.*}}i32 @foo_mma.default()
// CHECK: define internal ptr @foo_mma.resolver()
//   if (__builtin_cpu_supports("mma")) return &foo_mma.mma;
// CHECK: %[[#MMA:]] = call i64 @getsystemcfg(i32 62)
// CHECK-NEXT: icmp ugt i64 %[[#MMA]], 0
// CHECK: ret ptr @foo_mma.mma
// CHECK: ret ptr @foo_mma.default

// CHECK: declare i64 @getsystemcfg(i32)
