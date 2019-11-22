// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=50 -DGPU
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -fopenmp-version=50 -DGPU | FileCheck %s --implicit-check-not='ret i32 {{1|81|84}}'
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t -fopenmp-version=50 -DGPU
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - -fopenmp-version=50 -DGPU | FileCheck %s --implicit-check-not='ret i32 {{1|81|84}}'

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=50 -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -fopenmp-version=50 -DNOHOST | FileCheck %s --implicit-check-not='ret i32 {{1|81|84}}'
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t -fopenmp-version=50 -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - -fopenmp-version=50 -DNOHOST | FileCheck %s --implicit-check-not='ret i32 {{1|81|84}}'
// expected-no-diagnostics

// CHECK-NOT: ret i32 {{1|81|84}}
// CHECK-DAG: define {{.*}}i32 @_Z3barv()
// CHECK-DAG: define {{.*}}i32 @_ZN16SpecSpecialFuncs6MethodEv(%struct.SpecSpecialFuncs* %{{.+}})
// CHECK-DAG: define {{.*}}i32 @_ZN12SpecialFuncs6MethodEv(%struct.SpecialFuncs* %{{.+}})
// CHECK-DAG: define linkonce_odr {{.*}}i32 @_ZN16SpecSpecialFuncs6methodEv(%struct.SpecSpecialFuncs* %{{.+}})
// CHECK-DAG: define linkonce_odr {{.*}}i32 @_ZN12SpecialFuncs6methodEv(%struct.SpecialFuncs* %{{.+}})
// CHECK-DAG: define {{.*}}i32 @_Z5prio_v()
// CHECK-DAG: define internal i32 @_ZL6prio1_v()
// CHECK-DAG: define {{.*}}i32 @_Z4callv()
// CHECK-DAG: define internal i32 @_ZL9stat_usedv()
// CHECK-DAG: define {{.*}}i32 @fn_linkage()
// CHECK-DAG: define {{.*}}i32 @_Z11fn_linkage1v()

// CHECK-DAG: ret i32 2
// CHECK-DAG: ret i32 3
// CHECK-DAG: ret i32 4
// CHECK-DAG: ret i32 5
// CHECK-DAG: ret i32 6
// CHECK-DAG: ret i32 7
// CHECK-DAG: ret i32 82
// CHECK-DAG: ret i32 83
// CHECK-DAG: ret i32 85
// CHECK-DAG: ret i32 86
// CHECK-DAG: ret i32 87

// Outputs for function members
// CHECK-DAG: ret i32 6
// CHECK-DAG: ret i32 7
// CHECK-NOT: ret i32 {{1|81|84}}

#ifndef HEADER
#define HEADER

#ifdef GPU
#define CORRECT gpu
#define SUBSET nohost, gpu
#define WRONG cpu, gpu
#endif // GPU
#ifdef NOHOST
#define CORRECT nohost
#define SUBSET nohost, gpu
#define WRONG nohost, host
#endif // NOHOST

int foo() { return 2; }
int bazzz();
int test();
static int stat_unused_();
static int stat_used_();

#pragma omp declare target

#pragma omp declare variant(foo) match(device = {kind(CORRECT)})
int bar() { return 1; }

#pragma omp declare variant(bazzz) match(device = {kind(CORRECT)})
int baz() { return 1; }

#pragma omp declare variant(test) match(device = {kind(CORRECT)})
int call() { return 1; }

#pragma omp declare variant(stat_unused_) match(device = {kind(CORRECT)})
static int stat_unused() { return 1; }

#pragma omp declare variant(stat_used_) match(device = {kind(CORRECT)})
static int stat_used() { return 1; }

#pragma omp end declare target

int main() {
  int res;
#pragma omp target map(from \
                       : res)
  res = bar() + baz() + call();
  return res;
}

int test() { return 3; }
static int stat_unused_() { return 4; }
static int stat_used_() { return 5; }

#pragma omp declare target

struct SpecialFuncs {
  void vd() {}
  SpecialFuncs();
  ~SpecialFuncs();

  int method_() { return 6; }
#pragma omp declare variant(SpecialFuncs::method_) \
    match(device = {kind(CORRECT)})
  int method() { return 1; }
#pragma omp declare variant(SpecialFuncs::method_) \
    match(device = {kind(CORRECT)})
  int Method();
} s;

int SpecialFuncs::Method() { return 1; }

struct SpecSpecialFuncs {
  void vd() {}
  SpecSpecialFuncs();
  ~SpecSpecialFuncs();

  int method_();
#pragma omp declare variant(SpecSpecialFuncs::method_) \
    match(device = {kind(CORRECT)})
  int method() { return 1; }
#pragma omp declare variant(SpecSpecialFuncs::method_) \
    match(device = {kind(CORRECT)})
  int Method();
} s1;

#pragma omp end declare target

int SpecSpecialFuncs::method_() { return 7; }
int SpecSpecialFuncs::Method() { return 1; }

int prio() { return 81; }
int prio1() { return 82; }
static int prio2() { return 83; }
static int prio3() { return 84; }
static int prio4() { return 84; }
int fn_linkage_variant() { return 85; }
extern "C" int fn_linkage_variant1() { return 86; }
int fn_variant2() { return 1; }

#pragma omp declare target

void xxx() {
  (void)s.method();
  (void)s1.method();
}

#pragma omp declare variant(prio) match(device = {kind(SUBSET)})
#pragma omp declare variant(prio1) match(device = {kind(CORRECT)})
int prio_() { return 1; }

#pragma omp declare variant(prio4) match(device = {kind(SUBSET)})
#pragma omp declare variant(prio2) match(device = {kind(CORRECT)})
#pragma omp declare variant(prio3) match(device = {kind(SUBSET)})
static int prio1_() { return 1; }

int int_fn() { return prio1_(); }

extern "C" {
#pragma omp declare variant(fn_linkage_variant) match(device = {kind(CORRECT)})
int fn_linkage() { return 1; }
}

#pragma omp declare variant(fn_linkage_variant1) match(device = {kind(CORRECT)})
int fn_linkage1() { return 1; }

#pragma omp declare variant(fn_variant2) match(device = {kind(WRONG)})
int fn2() { return 87; }

#pragma omp end declare target

#endif // HEADER
