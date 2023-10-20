// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DHOST | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t -fopenmp-version=45 %s -DHOST
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 -DHOST | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple aarch64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DHOST | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple aarch64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t -fopenmp-version=45 %s -DHOST
// RUN: %clang_cc1 -fopenmp -x c++ -triple aarch64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 -DHOST | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DHOST | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple ppc64le-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t -fopenmp-version=45 %s -DHOST
// RUN: %clang_cc1 -fopenmp -x c++ -triple ppc64le-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 -DHOST | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DCPU | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t -fopenmp-version=45 %s -DCPU
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 -DCPU | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple aarch64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DCPU | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple aarch64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t -fopenmp-version=45 %s -DCPU
// RUN: %clang_cc1 -fopenmp -x c++ -triple aarch64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 -DCPU | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DCPU | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple ppc64le-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t -fopenmp-version=45 %s -DCPU
// RUN: %clang_cc1 -fopenmp -x c++ -triple ppc64le-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 -DCPU | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux -fopenmp-targets=x86_64-unknown-linux -emit-llvm-bc %s -o %t-host.bc -DCPU
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - -DCPU | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-pch -o %t -DCPU
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -include-pch %t -o - -DCPU | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple ppc64le-unknown-linux -fopenmp-targets=ppc64le-unknown-linux -emit-llvm-bc %s -o %t-host.bc -DCPU
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - -DCPU | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-pch -o %t -DCPU
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -include-pch %t -o - -DCPU | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux -fopenmp-targets=x86_64-unknown-linux -emit-llvm-bc %s -o %t-host.bc -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - -DNOHOST | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-pch -o %t -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -include-pch %t -o - -DNOHOST | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple ppc64le-unknown-linux -fopenmp-targets=ppc64le-unknown-linux -emit-llvm-bc %s -o %t-host.bc -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - -DNOHOST | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-pch -o %t -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -include-pch %t -o - -DNOHOST | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DHOST | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s -DHOST
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -DHOST | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple aarch64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DHOST | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple aarch64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s -DHOST
// RUN: %clang_cc1 -fopenmp -x c++ -triple aarch64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -DHOST | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DHOST | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple ppc64le-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s -DHOST
// RUN: %clang_cc1 -fopenmp -x c++ -triple ppc64le-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -DHOST | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DCPU | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s -DCPU
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -DCPU | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple aarch64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DCPU | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple aarch64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s -DCPU
// RUN: %clang_cc1 -fopenmp -x c++ -triple aarch64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -DCPU | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - -fsanitize-address-use-after-scope -DCPU | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple ppc64le-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s -DCPU
// RUN: %clang_cc1 -fopenmp -x c++ -triple ppc64le-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -DCPU | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=x86_64-unknown-linux -emit-llvm-bc %s -o %t-host.bc -DCPU
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - -DCPU | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-pch -o %t -DCPU
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -include-pch %t -o - -DCPU | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -fopenmp-targets=ppc64le-unknown-linux -emit-llvm-bc %s -o %t-host.bc -DCPU
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - -DCPU | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-pch -o %t -DCPU
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -include-pch %t -o - -DCPU | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=x86_64-unknown-linux -emit-llvm-bc %s -o %t-host.bc -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - -DNOHOST | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-pch -o %t -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -include-pch %t -o - -DNOHOST | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -fopenmp-targets=ppc64le-unknown-linux -emit-llvm-bc %s -o %t-host.bc -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - -DNOHOST | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-pch -o %t -DNOHOST
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -include-pch %t -o - -DNOHOST | FileCheck %s

// expected-no-diagnostics

// CHECK-NOT: {{ (no)?alias }}

// CHECK-NOT: ret i32 {{1|4|81|84}}
// CHECK-DAG: declare {{.*}}i32 @_Z5bazzzv()
// CHECK-DAG: define {{.*}}i32 @_Z3bazv()
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
// CHECK-NOT: ret i32 {{4|81|84}}

#ifndef HEADER
#define HEADER

#pragma omp declare target
#ifdef HOST
#define SUBSET host
#define CORRECT host, cpu
#define WRONG host, nohost
#endif // HOST
#ifdef CPU
#define SUBSET cpu
#define CORRECT cpu, any
#define WRONG cpu, gpu
#endif // CPU
#ifdef NOHOST
#define SUBSET nohost
#define CORRECT nohost, cpu
#define WRONG nohost, host
#endif // NOHOST

int foo() { return 2; }

#pragma omp declare variant(foo) match(device = {kind(CORRECT)})
int bar() { return 1; }

int bazzz();
#pragma omp declare variant(bazzz) match(device = {kind(CORRECT)})
int baz() { return 1; }

int test();
#pragma omp declare variant(test) match(device = {kind(CORRECT)})
int call() { return 1; }

static int stat_unused_();
#pragma omp declare variant(stat_unused_) match(device = {kind(CORRECT)})
static int stat_unused() { return 1; }

static int stat_used_();
#pragma omp declare variant(stat_used_) match(device = {kind(CORRECT)})
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

int SpecSpecialFuncs::method_() { return 7; }
int SpecSpecialFuncs::Method() { return 1; }

void xxx() {
  (void)s.method();
  (void)s1.method();
}

int prio() { return 81; }
int prio1() { return 82; }

#pragma omp declare variant(prio) match(device = {kind(SUBSET)})
#pragma omp declare variant(prio1) match(device = {kind(CORRECT)})
int prio_() { return 1; }

static int prio2() { return 83; }
static int prio3() { return 84; }
static int prio4() { return 84; }

#pragma omp declare variant(prio4) match(device = {kind(SUBSET)})
#pragma omp declare variant(prio2) match(device = {kind(CORRECT)})
#pragma omp declare variant(prio3) match(device = {kind(SUBSET)})
static int prio1_() { return 1; }

int int_fn() { return prio1_(); }

int fn_linkage_variant() { return 85; }
extern "C" {
#pragma omp declare variant(fn_linkage_variant) match(device = {kind(CORRECT)})
int fn_linkage() { return 1; }
}

extern "C" int fn_linkage_variant1() { return 86; }
#pragma omp declare variant(fn_linkage_variant1) match(device = {kind(CORRECT)})
int fn_linkage1() { return 1; }

int fn_variant2() { return 1; }
#pragma omp declare variant(fn_variant2) match(device = {kind(WRONG)})
int fn2() { return 87; }

#pragma omp end declare target
#endif // HEADER
