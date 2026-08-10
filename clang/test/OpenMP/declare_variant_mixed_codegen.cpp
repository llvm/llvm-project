// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|10|13|15|19|22|23|24}}'
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|10|13|15|19|22|23|24}}'

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|10|13|15|19|22|23|24}}'
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|10|13|15|19|22|23|24}}'

// expected-no-diagnostics

// CHECK-DAG:  ret i32 2
// CHECK-DAG:  ret i32 3
// CHECK-DAG:  ret i32 4
// CHECK-DAG:  ret i32 5
// CHECK-DAG:  ret i32 9
// CHECK-DAG:  ret i32 11
// CHECK-DAG:  ret i32 12
// CHECK-DAG:  ret i32 14
// CHECK-DAG:  ret i32 16
// CHECK-DAG:  ret i32 17
// CHECK-DAG:  ret i32 18
// CHECK-DAG:  ret i32 19
// CHECK-DAG:  ret i32 20
// CHECK-DAG:  ret i32 21
// CHECK-DAG:  ret i32 25
// CHECK-DAG:  ret i32 26
// CHECK-DAG:  ret i32 27
// CHECK-DAG:  ret i32 28
// CHECK-DAG:  ret i32 29
// CHECK-DAG:  ret i32 30
// CHECK-DAG:  ret i32 31

#ifndef HEADER
#define HEADER

int foo() { return 2; }

#pragma omp declare variant(foo) match(implementation = {vendor(llvm)}, device={kind(cpu)})
int bar() { return 3; }

int bazzz();
#pragma omp declare variant(bazzz) match(implementation = {vendor(llvm)}, device={kind(host)})
int baz() { return 4; }

int test();
#pragma omp declare variant(test) match(implementation = {vendor(llvm)}, device={kind(cpu)})
int call() { return 5; }

static int stat_unused_no_emit() { return 6; }
static int stat_unused_();
#pragma omp declare variant(stat_unused_) match(implementation = {vendor(llvm)}, device={kind(cpu)})
#pragma omp declare variant(stat_unused_no_emit) match(implementation = {vendor(unknown)}, device = {kind(gpu)})
static int stat_unused() { return 7; }

static int stat_used_();
#pragma omp declare variant(stat_used_) match(implementation = {vendor(llvm)}, device={kind(host)})
static int stat_used() { return 8; }

int main() { return bar() + baz() + call() + stat_used(); }

int test() { return 9; }
static int stat_unused_() { return 10; }
static int stat_used_() { return 11; }

struct SpecialFuncs {
  void vd() {}
  SpecialFuncs();
  ~SpecialFuncs();

  int method_() { return 12; }
#pragma omp declare variant(SpecialFuncs::method_)                             \
    match(implementation = {vendor(llvm)}, device={kind(cpu)})
  int method() { return 13; }
#pragma omp declare variant(SpecialFuncs::method_)                             \
    match(implementation = {vendor(llvm)}, device={kind(host)})
  int Method();
} s;

int SpecialFuncs::Method() { return 14; }

struct SpecSpecialFuncs {
  void vd() {}
  SpecSpecialFuncs();
  ~SpecSpecialFuncs();

  int method_();
#pragma omp declare variant(SpecSpecialFuncs::method_)                         \
    match(implementation = {vendor(llvm)}, device={kind(cpu)})
  int method() { return 15; }
#pragma omp declare variant(SpecSpecialFuncs::method_)                         \
    match(implementation = {vendor(llvm)}, device={kind(host)})
  int Method();
} s1;

int SpecSpecialFuncs::method_() { return 16; }
int SpecSpecialFuncs::Method() { return 17; }

void xxx() {
  (void)s.method();
  (void)s1.method();
}

int prio() { return 18; }
int prio1() { return 19; }

#pragma omp declare variant(prio1) match(implementation = {vendor(score(2): llvm)}, device={kind(cpu,host)})
#pragma omp declare variant(prio) match(implementation = {vendor(score(1): llvm)}, device={kind(cpu)})
int prio_() { return 20; }

static int prio2() { return 21; }
static int prio3() { return 22; }
static int prio4() { return 23; }

#pragma omp declare variant(prio4) match(implementation = {vendor(score(5): llvm)})
#pragma omp declare variant(prio2) match(implementation = {vendor(score(8): llvm)}, device={kind(cpu,host)})
#pragma omp declare variant(prio3) match(implementation = {vendor(score(7): llvm)}, device={kind(cpu)})
static int prio1_() { return 24; }

int int_fn() { return prio1_(); }

int fn_linkage_variant() { return 25; }
extern "C" {
#pragma omp declare variant(fn_linkage_variant) match(implementation = {vendor(llvm)}, device={kind(cpu)})
int fn_linkage() { return 26; }
}

extern "C" int fn_linkage_variant1() { return 27; }
#pragma omp declare variant(fn_linkage_variant1) match(implementation = {vendor(llvm)}, device={kind(host)})
int fn_linkage1() { return 28; }

int fn_variant2() { return 29; }
#pragma omp declare variant(fn_variant2) match(implementation = {vendor(llvm, ibm)}, device={kind(cpu)})
#pragma omp declare variant(fn_variant2) match(implementation = {vendor(llvm)}, device={kind(cpu,gpu)})
#pragma omp declare variant(fn_variant2) match(implementation = {vendor(llvm)}, device={kind(nohost)})
#pragma omp declare variant(fn_variant2) match(implementation = {vendor(llvm)}, device={kind(cpu,nohost)})
#pragma omp declare variant(fn_variant2) match(implementation = {vendor(llvm)}, device={kind(gpu)})
#pragma omp declare variant(fn_variant2) match(implementation = {vendor(llvm)}, device={kind(fpga)})
int fn2() { return 30; }

#pragma omp declare variant(stat_unused_no_emit) match(implementation = {vendor(unknown)}, device = {kind(gpu)})
template <typename T>
static T stat_unused_T() { return 31; }

int bazzzzzzzz() {
  return stat_unused_T<int>();
}

#endif // HEADER
