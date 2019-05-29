// RUN: %clang_cc1 -x c++ -triple spir64-unknown-linux-sycldevice -std=c++11 -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// CHECK: br label %for.cond, !llvm.loop ![[MD_A:[0-9]+]]
// CHECK: br label %for.cond, !llvm.loop ![[MD_B:[0-9]+]]
// CHECK: br label %for.cond, !llvm.loop ![[MD_C:[0-9]+]]
// CHECK: br label %for.cond, !llvm.loop ![[MD_D:[0-9]+]]

// CHECK: ![[MD_A]] = distinct !{![[MD_A]], ![[MD_ivdep_A:[0-9]+]]}
// CHECK-NEXT: ![[MD_ivdep_A]] = !{!"llvm.loop.ivdep.enable"}
void foo() {
  int a[10];
  [[intelfpga::ivdep]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

// CHECK: ![[MD_B]] = distinct !{![[MD_B]], ![[MD_ivdep_B:[0-9]+]]}
// CHECK-NEXT: ![[MD_ivdep_B]] = !{!"llvm.loop.ivdep.safelen", i32 2}
void boo() {
  int a[10];
  [[intelfpga::ivdep(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

// CHECK: ![[MD_C]] = distinct !{![[MD_C]], ![[MD_ii:[0-9]+]]}
// CHECK-NEXT: ![[MD_ii]] = !{!"llvm.loop.ii.count", i32 2}
void goo() {
  int a[10];
  [[intelfpga::ii(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

// CHECK: ![[MD_D]] = distinct !{![[MD_D]], ![[MD_max_concurrency:[0-9]+]]}
// CHECK-NEXT: ![[MD_max_concurrency]] = !{!"llvm.loop.max_concurrency.count", i32 2}
void zoo() {
  int a[10];
  [[intelfpga::max_concurrency(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo();
    boo();
    goo();
    zoo();
  });
  return 0;
}
