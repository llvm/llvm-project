// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple %itanium_abi_triple -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -fopenmp-version=45 | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|9|10|12|13|14|15|19|21|22|23|24}}'
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|9|10|12|13|14|15|19|21|22|23|24}}'
// expected-no-diagnostics

// CHECK-DAG:  ret i32 2
// CHECK-DAG:  ret i32 3
// CHECK-DAG:  ret i32 4
// CHECK-DAG:  ret i32 5
// CHECK-DAG:  ret i32 11
// CHECK-DAG:  ret i32 16
// CHECK-DAG:  ret i32 17
// CHECK-DAG:  ret i32 18
// CHECK-DAG:  ret i32 19
// CHECK-DAG:  ret i32 20
// CHECK-DAG:  ret i32 25
// CHECK-DAG:  ret i32 26
// CHECK-DAG:  ret i32 27
// CHECK-DAG:  ret i32 28
// CHECK-DAG:  ret i32 29

#ifndef HEADER
#define HEADER

int foo() { return 2; }

#pragma omp declare variant(foo) match(implementation = {vendor(amd)})
int bar() { return 3; }

int bazzz();
#pragma omp declare variant(bazzz) match(implementation = {vendor(amd)})
int baz() { return 4; }

int test();
#pragma omp declare variant(test) match(implementation = {vendor(amd)})
int call() { return 5; }

static int stat_unused_();
#pragma omp declare variant(stat_unused_) match(implementation = {vendor(amd)})
static int stat_unused() { return 6; }

static int stat_used_();
#pragma omp declare variant(stat_used_) match(implementation = {vendor(amd)})
static int stat_used() { return 7; }

int main() { return bar() + baz() + call() + stat_used(); }

int test() { return 8; }
static int stat_unused_() { return 9; }
static int stat_used_() { return 10; }

struct SpecialFuncs {
  void vd() {}
  SpecialFuncs();
  ~SpecialFuncs();

  int method_() { return 11; }
#pragma omp declare variant(SpecialFuncs::method_)                             \
    match(implementation = {vendor(amd)})
  int method() { return 12; }
#pragma omp declare variant(SpecialFuncs::method_)                             \
    match(implementation = {vendor(amd)})
  int Method();
} s;

int SpecialFuncs::Method() { return 13; }

struct SpecSpecialFuncs {
  void vd() {}
  SpecSpecialFuncs();
  ~SpecSpecialFuncs();

  int method_();
#pragma omp declare variant(SpecSpecialFuncs::method_)                         \
    match(implementation = {vendor(amd)})
  int method() { return 14; }
#pragma omp declare variant(SpecSpecialFuncs::method_)                         \
    match(implementation = {vendor(amd)})
  int Method();
} s1;

int SpecSpecialFuncs::method_() { return 15; }
int SpecSpecialFuncs::Method() { return 16; }

void xxx() {
  (void)s.method();
  (void)s1.method();
}

int prio() { return 17; }
int prio1() { return 18; }

#pragma omp declare variant(prio) match(implementation = {vendor(amd)})
#pragma omp declare variant(prio1) match(implementation = {vendor(score(1): amd)})
int prio_() { return 19; }

static int prio2() { return 20; }
static int prio3() { return 21; }
static int prio4() { return 22; }

#pragma omp declare variant(prio4) match(implementation = {vendor(score(3): amd)})
#pragma omp declare variant(prio2) match(implementation = {vendor(score(5): amd)})
#pragma omp declare variant(prio3) match(implementation = {vendor(score(1): amd)})
static int prio1_() { return 23; }

int int_fn() { return prio1_(); }

int fn_linkage_variant() { return 24; }
extern "C" {
#pragma omp declare variant(fn_linkage_variant) match(implementation = {vendor(amd)})
int fn_linkage() { return 25; }
}

extern "C" int fn_linkage_variant1() { return 26; }
#pragma omp declare variant(fn_linkage_variant1) match(implementation = {vendor(amd)})
int fn_linkage1() { return 27; }

int fn_variant2() { return 28; }
#pragma omp declare variant(fn_variant2) match(implementation = {vendor(amd, ibm)})
int fn2() { return 29; }

#endif // HEADER
