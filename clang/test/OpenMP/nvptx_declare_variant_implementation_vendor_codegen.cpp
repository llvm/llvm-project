// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=45
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -fopenmp-version=45 | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|9|10|12|13|14|15|17|18|19|20|21|22|23|24|26}}'
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t -fopenmp-version=45
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - -fopenmp-version=45 | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|9|10|12|13|14|15|17|18|19|20|21|22|23|24|26}}'

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|9|10|12|13|14|15|17|18|19|20|21|22|23|24|26}}'
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --implicit-check-not='ret i32 {{6|7|8|9|10|12|13|14|15|17|18|19|20|21|22|23|24|26}}'
// expected-no-diagnostics

// CHECK-DAG:  ret i32 2
// CHECK-DAG:  ret i32 3
// CHECK-DAG:  ret i32 4
// CHECK-DAG:  ret i32 5
// CHECK-DAG:  ret i32 11
// CHECK-DAG:  ret i32 16
// CHECK-DAG:  ret i32 19
// CHECK-DAG:  ret i32 25
// CHECK-DAG:  ret i32 27
// CHECK-DAG:  ret i32 28
// CHECK-DAG:  ret i32 29

#ifndef HEADER
#define HEADER

int foo() { return 2; }
int bazzz();
int test();
static int stat_unused_();
static int stat_used_();

#pragma omp declare target

#pragma omp declare variant(foo) match(implementation = {vendor(amd)})
int bar() { return 3; }

#pragma omp declare variant(bazzz) match(implementation = {vendor(amd)})
int baz() { return 4; }

#pragma omp declare variant(test) match(implementation = {vendor(amd)})
int call() { return 5; }

#pragma omp declare variant(stat_unused_) match(implementation = {vendor(amd)})
static int stat_unused() { return 6; }

#pragma omp declare variant(stat_used_) match(implementation = {vendor(amd)})
static int stat_used() { return 7; }

#pragma omp end declare target

int main() {
  int res;
#pragma omp target map(from \
                       : res)
  res = bar() + baz() + call();
  return res;
}

int test() { return 8; }
static int stat_unused_() { return 9; }
static int stat_used_() { return 10; }

#pragma omp declare target

struct SpecialFuncs {
  void vd() {}
  SpecialFuncs();
  ~SpecialFuncs();

  int method_() { return 11; }
#pragma omp declare variant(SpecialFuncs::method_) \
    match(implementation = {vendor(amd)})
  int method() { return 12; }
#pragma omp declare variant(SpecialFuncs::method_) \
    match(implementation = {vendor(amd)})
  int Method();
} s;

int SpecialFuncs::Method() { return 13; }

struct SpecSpecialFuncs {
  void vd() {}
  SpecSpecialFuncs();
  ~SpecSpecialFuncs();

  int method_();
#pragma omp declare variant(SpecSpecialFuncs::method_) \
    match(implementation = {vendor(amd)})
  int method() { return 14; }
#pragma omp declare variant(SpecSpecialFuncs::method_) \
    match(implementation = {vendor(amd)})
  int Method();
} s1;

#pragma omp end declare target

int SpecSpecialFuncs::method_() { return 15; }
int SpecSpecialFuncs::Method() { return 16; }

int prio() { return 17; }
int prio1() { return 18; }
static int prio2() { return 19; }
static int prio3() { return 20; }
static int prio4() { return 21; }
int fn_linkage_variant() { return 22; }
extern "C" int fn_linkage_variant1() { return 23; }
int fn_variant2() { return 24; }

#pragma omp declare target

void xxx() {
  (void)s.method();
  (void)s1.method();
}

#pragma omp declare variant(prio) match(implementation = {vendor(amd)})
#pragma omp declare variant(prio1) match(implementation = {vendor(score(1) \
                                                                  : amd)})
int prio_() { return 25; }

#pragma omp declare variant(prio4) match(implementation = {vendor(score(3) \
                                                                  : amd)})
#pragma omp declare variant(prio2) match(implementation = {vendor(score(5) \
                                                                  : amd)})
#pragma omp declare variant(prio3) match(implementation = {vendor(score(1) \
                                                                  : amd)})
static int prio1_() { return 26; }

int int_fn() { return prio1_(); }

extern "C" {
#pragma omp declare variant(fn_linkage_variant) match(implementation = {vendor(amd)})
int fn_linkage() { return 27; }
}

#pragma omp declare variant(fn_linkage_variant1) match(implementation = {vendor(amd)})
int fn_linkage1() { return 28; }

#pragma omp declare variant(fn_variant2) match(implementation = {vendor(llvm, ibm)})
int fn2() { return 29; }

#pragma omp end declare target

#endif // HEADER
