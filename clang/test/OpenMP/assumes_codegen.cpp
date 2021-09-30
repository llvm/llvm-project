// RUN: %clang_cc1 -verify -fopenmp -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s --check-prefix=AST
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -verify -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify=pch %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-enable-irbuilder -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -verify -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify=pch %s -emit-llvm -o - | FileCheck %s

// pch-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {
}

#pragma omp assumes no_openmp_routines warning ext_another_warning(1) ext_after_invalid_clauses // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}} expected-warning {{'ext_another_warning' clause should not be followed by arguments; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}

#pragma omp assumes no_openmp

#pragma omp begin assumes ext_range_bar_only

#pragma omp begin assumes ext_range_bar_only_2

class BAR {
public:
  BAR() {}

  void bar1() {
  }

  static void bar2() {
  }
};

void bar() { BAR b; }

#pragma omp end assumes
#pragma omp end assumes

#pragma omp begin assumes ext_not_seen
#pragma omp end assumes

#pragma omp begin assumes ext_1234
void baz();

template<typename T>
class BAZ {
public:
  BAZ() {}

  void baz1() {
  }

  static void baz2() {
  }
};

void baz() { BAZ<float> b; }
#pragma omp end assumes

#pragma omp begin assumes ext_lambda_assumption
int lambda_outer() {
  auto lambda_inner = []() { return 42; };
  return lambda_inner();
}
#pragma omp end assumes

void no_assume() {
  foo();
}

#pragma omp begin assumes ext_call_site
void assume() {
  foo();
}

void assembly() {
  asm ("nop");
}
#pragma omp end assumes ext_call_site

// AST:      void foo() __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp"))) {
// AST-NEXT: }
// AST-NEXT: class BAR {
// AST-NEXT: public:
// AST-NEXT:     BAR() __attribute__((assume("ompx_range_bar_only"))) __attribute__((assume("ompx_range_bar_only_2"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")))     {
// AST-NEXT:     }
// AST-NEXT:     void bar1() __attribute__((assume("ompx_range_bar_only"))) __attribute__((assume("ompx_range_bar_only_2"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")))     {
// AST-NEXT:     }
// AST-NEXT:     static void bar2() __attribute__((assume("ompx_range_bar_only"))) __attribute__((assume("ompx_range_bar_only_2"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")))     {
// AST-NEXT:     }
// AST-NEXT: };
// AST-NEXT: void bar() __attribute__((assume("ompx_range_bar_only"))) __attribute__((assume("ompx_range_bar_only_2"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp"))) {
// AST-NEXT:     BAR b;
// AST-NEXT: }
// AST-NEXT: void baz() __attribute__((assume("ompx_1234"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")));
// AST-NEXT: template <typename T> class BAZ {
// AST-NEXT: public:
// AST-NEXT:     BAZ<T>() __attribute__((assume("ompx_1234"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")))     {
// AST-NEXT:     }
// AST-NEXT:     void baz1() __attribute__((assume("ompx_1234"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")))     {
// AST-NEXT:     }
// AST-NEXT:     static void baz2() __attribute__((assume("ompx_1234"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")))     {
// AST-NEXT:     }
// AST-NEXT: };
// AST-NEXT: template<> class BAZ<float> {
// AST-NEXT: public:
// AST-NEXT:     BAZ() __attribute__((assume("ompx_1234"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")))    {
// AST-NEXT:     }
// AST-NEXT:     void baz1() __attribute__((assume("ompx_1234"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")));
// AST-NEXT:     static void baz2() __attribute__((assume("ompx_1234"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp")));
// AST-NEXT: };
// AST-NEXT: void baz() __attribute__((assume("ompx_1234"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp"))) {
// AST-NEXT:     BAZ<float> b;
// AST-NEXT: }
// AST-NEXT: int lambda_outer() __attribute__((assume("ompx_lambda_assumption"))) __attribute__((assume("omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses"))) __attribute__((assume("omp_no_openmp"))) {
// AST-NEXT:     auto lambda_inner = []() {
// AST-NEXT:         return 42;
// AST-NEXT:     };
// AST-NEXT:     return lambda_inner();
// AST-NEXT: }

#endif

// CHECK: define{{.*}} void @_Z3foov()
// CHECK-SAME: [[attr0:#[0-9]]]
// CHECK: define{{.*}} void @_Z3barv()
// CHECK-SAME: [[attr1:#[0-9]]]
// CHECK:   call{{.*}} @_ZN3BARC1Ev(%class.BAR*{{.*}} %b)
// CHECK-SAME: [[attr10:#[0-9]]]
// CHECK: define{{.*}} void @_ZN3BARC1Ev(%class.BAR*{{.*}} %this)
// CHECK-SAME: [[attr2:#[0-9]]]
// CHECK:   call{{.*}} @_ZN3BARC2Ev(%class.BAR*{{.*}} %this1)
// CHECK-SAME: [[attr10]]
// CHECK: define{{.*}} void @_ZN3BARC2Ev(%class.BAR*{{.*}} %this)
// CHECK-SAME: [[attr3:#[0-9]]]
// CHECK: define{{.*}} void @_Z3bazv()
// CHECK-SAME: [[attr4:#[0-9]]]
// CHECK:   call{{.*}} @_ZN3BAZIfEC1Ev(%class.BAZ*{{.*}} %b)
// CHECK-SAME: [[attr11:#[0-9]]]
// CHECK: define{{.*}} void @_ZN3BAZIfEC1Ev(%class.BAZ*{{.*}} %this)
// CHECK-SAME: [[attr5:#[0-9]]]
// CHECK:   call{{.*}} @_ZN3BAZIfEC2Ev(%class.BAZ*{{.*}} %this1)
// CHECK-SAME: [[attr12:#[0-9]]]
// CHECK: define{{.*}} void @_ZN3BAZIfEC2Ev(%class.BAZ*{{.*}} %this)
// CHECK-SAME: [[attr6:#[0-9]]]
// CHECK: define{{.*}} i32 @_Z12lambda_outerv()
// CHECK-SAME: [[attr7:#[0-9]]]
// CHECK: call{{.*}} @"_ZZ12lambda_outervENK3$_0clEv"
// CHECK-SAME: [[attr13:#[0-9]]]
// CHECK: define{{.*}} i32 @"_ZZ12lambda_outervENK3$_0clEv"(%class.anon*{{.*}} %this)
// CHECK-SAME: [[attr8:#[0-9]]]
// CHECK: define{{.*}} void @_Z9no_assumev()
// CHECK-SAME: [[attr0:#[0-9]]]
// CHECK: call{{.*}} @_Z3foov()
// CHECK-SAME: [[attr14:#[0-9]]]
// CHECK: define{{.*}} void @_Z6assumev()
// CHECK-SAME: [[attr9:#[0-9]]]
// CHECK: call{{.*}} @_Z3foov()
// CHECK-SAME: [[attr15:#[0-9]]]
// CHECK: define{{.*}} void @_Z8assemblyv()
// CHECK-SAME: [[attr9:#[0-9]]]
// CHECK: call{{.*}} void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
// CHECK-SAME: [[attr16:#[0-9]]]

// CHECK:     attributes [[attr0]]
// CHECK-SAME:  "llvm.assume"="omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr1]]
// CHECK-SAME:  "llvm.assume"="ompx_range_bar_only,ompx_range_bar_only_2,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr2]]
// CHECK-SAME:  "llvm.assume"="ompx_range_bar_only,ompx_range_bar_only_2,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr3]]
// CHECK-SAME:  "llvm.assume"="ompx_range_bar_only,ompx_range_bar_only_2,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr4]]
// CHECK-SAME:  "llvm.assume"="ompx_1234,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,ompx_1234,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr5]]
// CHECK-SAME:  "llvm.assume"="ompx_1234,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr6]]
// CHECK-SAME:  "llvm.assume"="ompx_1234,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr7]]
// CHECK-SAME:  "llvm.assume"="ompx_lambda_assumption,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr8]]
// CHECK-SAME:  "llvm.assume"="ompx_lambda_assumption,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr9]]
// CHECK-SAME:  "llvm.assume"="ompx_call_site,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr10]]
// CHECK-SAME:  "llvm.assume"="ompx_range_bar_only,ompx_range_bar_only_2,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,ompx_range_bar_only,ompx_range_bar_only_2,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr11]]
// CHECK-SAME:  "llvm.assume"="ompx_1234,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,ompx_1234,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,ompx_1234,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr12]]
// CHECK-SAME:  "llvm.assume"="ompx_1234,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,ompx_1234,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr13]]
// CHECK-SAME:  "llvm.assume"="ompx_lambda_assumption,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,ompx_lambda_assumption,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr14]]
// CHECK-SAME:  "llvm.assume"="omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr15]]
// CHECK-SAME:  "llvm.assume"="omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp,ompx_call_site,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
// CHECK:     attributes [[attr16]]
// CHECK-SAME:  "llvm.assume"="ompx_call_site,omp_no_openmp_routines,ompx_another_warning,ompx_after_invalid_clauses,omp_no_openmp"
