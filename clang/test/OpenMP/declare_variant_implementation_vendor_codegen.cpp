// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple %itanium_abi_triple -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -emit-pch -o %t -fopenmp-version=50 %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=50 | FileCheck %s
// expected-no-diagnostics

// CHECK-NOT: ret i32 {{1|4|81|84}}
// CHECK-DAG: @_Z3barv = {{.*}}alias i32 (), i32 ()* @_Z3foov
// CHECK-DAG: @_ZN16SpecSpecialFuncs6MethodEv = {{.*}}alias i32 (%struct.SpecSpecialFuncs*), i32 (%struct.SpecSpecialFuncs*)* @_ZN16SpecSpecialFuncs7method_Ev
// CHECK-DAG: @_ZN16SpecSpecialFuncs6methodEv = linkonce_odr {{.*}}alias i32 (%struct.SpecSpecialFuncs*), i32 (%struct.SpecSpecialFuncs*)* @_ZN16SpecSpecialFuncs7method_Ev
// CHECK-DAG: @_ZN12SpecialFuncs6methodEv = linkonce_odr {{.*}}alias i32 (%struct.SpecialFuncs*), i32 (%struct.SpecialFuncs*)* @_ZN12SpecialFuncs7method_Ev
// CHECK-DAG: @_Z5prio_v = {{.*}}alias i32 (), i32 ()* @_Z5prio1v
// CHECK-DAG: @_ZL6prio1_v = internal alias i32 (), i32 ()* @_ZL5prio2v
// CHECK-DAG: @_Z4callv = {{.*}}alias i32 (), i32 ()* @_Z4testv
// CHECK-DAG: @_ZL9stat_usedv = internal alias i32 (), i32 ()* @_ZL10stat_used_v
// CHECK-DAG: @_ZN12SpecialFuncs6MethodEv = {{.*}}alias i32 (%struct.SpecialFuncs*), i32 (%struct.SpecialFuncs*)* @_ZN12SpecialFuncs7method_Ev
// CHECK-DAG: @fn_linkage = {{.*}}alias i32 (), i32 ()* @_Z18fn_linkage_variantv
// CHECK-DAG: @_Z11fn_linkage1v = {{.*}}alias i32 (), i32 ()* @fn_linkage_variant1
// CHECK-DAG: declare {{.*}}i32 @_Z5bazzzv()
// CHECK-DAG: declare {{.*}}i32 @_Z3bazv()
// CHECK-DAG: ret i32 2
// CHECK-DAG: ret i32 3
// CHECK-DAG: ret i32 5
// CHECK-DAG: ret i32 6
// CHECK-DAG: ret i32 7
// CHECK-DAG: ret i32 82
// CHECK-DAG: ret i32 83
// CHECK-DAG: ret i32 85
// CHECK-DAG: ret i32 86
// CHECK-DAG: ret i32 87
// CHECK-NOT: ret i32 {{1|4|81|84}}

#ifndef HEADER
#define HEADER

int foo() { return 2; }

#pragma omp declare variant(foo) match(implementation = {vendor(llvm)})
int bar() { return 1; }

int bazzz();
#pragma omp declare variant(bazzz) match(implementation = {vendor(llvm)})
int baz() { return 1; }

int test();
#pragma omp declare variant(test) match(implementation = {vendor(llvm)})
int call() { return 1; }

static int stat_unused_();
#pragma omp declare variant(stat_unused_) match(implementation = {vendor(llvm)})
static int stat_unused() { return 1; }

static int stat_used_();
#pragma omp declare variant(stat_used_) match(implementation = {vendor(llvm)})
static int stat_used() { return 1; }

int main() { return bar() + baz() + call() + stat_used(); }

int test() { return 3; }
static int stat_unused_() { return 4; }
static int stat_used_() { return 5; }

struct SpecialFuncs {
  void vd() {}
  SpecialFuncs();
  ~SpecialFuncs();

  int method_() { return 6; }
#pragma omp declare variant(SpecialFuncs::method_)                             \
    match(implementation = {vendor(llvm)})
  int method() { return 1; }
#pragma omp declare variant(SpecialFuncs::method_)                             \
    match(implementation = {vendor(llvm)})
  int Method();
} s;

int SpecialFuncs::Method() { return 1; }

struct SpecSpecialFuncs {
  void vd() {}
  SpecSpecialFuncs();
  ~SpecSpecialFuncs();

  int method_();
#pragma omp declare variant(SpecSpecialFuncs::method_)                         \
    match(implementation = {vendor(llvm)})
  int method() { return 1; }
#pragma omp declare variant(SpecSpecialFuncs::method_)                         \
    match(implementation = {vendor(llvm)})
  int Method();
} s1;

int SpecSpecialFuncs::method_() { return 7; }
int SpecSpecialFuncs::Method() { return 1; }

void xxx() {
  (void)s.method();
  (void)s1.method();
}

int prio() { return 81; }
int prio1() { return 82; }

#pragma omp declare variant(prio) match(implementation = {vendor(llvm)})
#pragma omp declare variant(prio1) match(implementation = {vendor(score(1): llvm)})
int prio_() { return 1; }

static int prio2() { return 83; }
static int prio3() { return 84; }
static int prio4() { return 84; }

#pragma omp declare variant(prio4) match(implementation = {vendor(score(3): llvm)})
#pragma omp declare variant(prio2) match(implementation = {vendor(score(5): llvm)})
#pragma omp declare variant(prio3) match(implementation = {vendor(score(1): llvm)})
static int prio1_() { return 1; }

int int_fn() { return prio1_(); }

int fn_linkage_variant() { return 85; }
extern "C" {
#pragma omp declare variant(fn_linkage_variant) match(implementation = {vendor(llvm)})
int fn_linkage() { return 1; }
}

extern "C" int fn_linkage_variant1() { return 86; }
#pragma omp declare variant(fn_linkage_variant1) match(implementation = {vendor(llvm)})
int fn_linkage1() { return 1; }

int fn_variant2() { return 1; }
#pragma omp declare variant(fn_variant2) match(implementation = {vendor(llvm, ibm)})
int fn2() { return 87; }

#endif // HEADER
