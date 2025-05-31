// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST

// expected-no-diagnostics
class C {
public:
//.
// HOST: @[[C:.+]] = internal global %class.C zeroinitializer, align 4
// HOST: @[[X:.+]] = internal global i32 0, align 4
// HOST: @y = hidden global i32 0
// HOST: @z = global i32 0
// HOST-NOT: @.offloading.entry.c
// HOST-NOT: @.offloading.entry.x
// HOST-NOT: @.offloading.entry.y
// HOST: @.offloading.entry.z
  C() : x(0) {}

  int x;
};

static C c;
#pragma omp declare target(c)

static int x;
#pragma omp declare target(x)

int __attribute__((visibility("hidden"))) y;
#pragma omp declare target(y)

int z;
#pragma omp declare target(z)
