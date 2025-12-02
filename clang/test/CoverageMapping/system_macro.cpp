// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -mllvm -system-headers-coverage=true -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm -main-file-name system_macro.cpp -o %t.w_sys.ll %s | FileCheck %s --check-prefixes=CHECK,W_SYS
// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -mllvm -system-headers-coverage=false -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm -main-file-name system_macro.cpp -o %t.wosys.ll %s | FileCheck %s --check-prefixes=CHECK,WOSYS
// RUN: FileCheck %s --check-prefixes=LL_CHECK,LL_W_SYS < %t.w_sys.ll
// RUN: FileCheck %s --check-prefixes=LL_CHECK,LL_WOSYS < %t.wosys.ll

// LL_CHECK: @__covrec_
// LL_W_SYS: [[PROFC:@.*__profc_.*SysTmpl.*]] =
// LL_W_SYS: @{{.*}}__profd_{{.*}}SysTmpl{{.*}} =
// LL_WOSYS-NOT: SysTmpl

// LL_CHECK: @llvm.used =

#ifdef IS_SYSHEADER

#pragma clang system_header
#define Func(x) if (x) {}
#define SomeType int

// LL_CHECK: define {{.*}} i1 @{{.*}}SysTmpl
template <bool f> bool SysTmpl() { return f; }
// Check SysTmpl() is instrumented or not.
// LL_W_SYS: load i64, ptr [[PROFC]],
// LL_WOSYS-NOT: load i64, ptr @__profc_

#else

#define IS_SYSHEADER
#include __FILE__

// CHECK-LABEL: doSomething
void doSomething(int x) { // CHECK: File 0, [[@LINE]]:25 -> {{[0-9:]+}} = #0
  // WOSYS-NOT: Expansion,
  // W_SYS: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:7
  Func(x);
  // CHECK: Gap,File 0, [[@LINE+1]]:10
  return;
  // WOSYS-NOT: Expansion,
  // W_SYS: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:11
  SomeType *f; // CHECK: File 0, [[@LINE]]:11 -> {{[0-9:]+}} = 0
}

// CHECK-LABEL: main
int main() { // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+2]]:2 = #0
  Func([] { return SysTmpl<true>(); }());
}

// W_SYS: SysTmpl
// WOSYS-NOT: SysTmpl

#endif
