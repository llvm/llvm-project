// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux -emit-llvm %s -o - \
// RUN:   | FileCheck %s

int v1(void);
int v2(void);

// MATCH_NONE accepts a construct selector outside that construct.
#pragma omp declare variant(v1) \
    match(construct = {parallel}, implementation = {extension(match_none)})
int none_base(void);

// CHECK-LABEL: define{{.*}} i32 @none_caller(
// CHECK: call i32 @v1()
int none_caller(void) { return none_base(); }

// Under MATCH_ANY, the matching construct adds its weight: 2 against 1.
#pragma omp declare variant(v1)                                               \
    match(construct = {parallel},                                             \
          implementation = {vendor(llvm), extension(match_any)})
#pragma omp declare variant(v2) match(user = {condition(1)})
int any_base(void);

// CHECK-LABEL: define{{.*}} @any_caller.omp_outlined(
// CHECK: call i32 @v1()
int any_caller(void) {
  int r = 0;
#pragma omp parallel
  r = any_base();
  return r;
}
