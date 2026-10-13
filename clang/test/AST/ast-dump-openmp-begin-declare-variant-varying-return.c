// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s        -DUSE_FLOAT | FileCheck %s --check-prefix=FLOAT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++ -DUSE_FLOAT | FileCheck %s --check-prefix=FLOAT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s                    | FileCheck %s --check-prefix=INT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++             | FileCheck %s --check-prefix=INT
// expected-no-diagnostics

#ifdef __cplusplus
#define OVERLOADABLE
#else
#define OVERLOADABLE __attribute__((overloadable))
#endif

#ifdef USE_FLOAT
#define RETURN_TY float
#define BEFORE_BASE_RETURN_VALUE 0
#define BEFORE_VARIANT_RETURN_VALUE 1
#define AFTER__BASE_RETURN_VALUE 1
#define AFTER__VARIANT_RETURN_VALUE 0
#else
#define RETURN_TY int
#define BEFORE_BASE_RETURN_VALUE 1
#define BEFORE_VARIANT_RETURN_VALUE 0
#define AFTER__BASE_RETURN_VALUE 0
#define AFTER__VARIANT_RETURN_VALUE 1
#endif

OVERLOADABLE
RETURN_TY also_before(void) {
  return BEFORE_BASE_RETURN_VALUE;
}
OVERLOADABLE
RETURN_TY also_before(int i) {
  return BEFORE_BASE_RETURN_VALUE;
}

#pragma omp begin declare variant match(implementation = {extension(disable_implicit_base)})
OVERLOADABLE
int also_before(void) {
  return BEFORE_VARIANT_RETURN_VALUE;
}
OVERLOADABLE
int also_before(int i) {
  return BEFORE_VARIANT_RETURN_VALUE;
}

OVERLOADABLE
int also_after(double d) {
  return AFTER__VARIANT_RETURN_VALUE;
}
OVERLOADABLE
int also_after(long l) {
  return AFTER__VARIANT_RETURN_VALUE;
}
#pragma omp end declare variant

OVERLOADABLE
RETURN_TY also_after(double d) {
  return AFTER__BASE_RETURN_VALUE;
}
OVERLOADABLE
RETURN_TY also_after(long l) {
  return AFTER__BASE_RETURN_VALUE;
}

int main(void) {
  // Should return 0.
  return also_before() + also_before(1) + also_before(2.0f) + also_after(3.0) + also_after(4L);
}

// Make sure we see base calls in the FLOAT versions.
// In the INT versions we want variant calls for the `*_before` functions
// but not the `*_after` ones (first 3 vs 2 last ones).

// FLOAT: call {{.*}} @{{.*}}also_beforev
// FLOAT: call {{.*}} @{{.*}}also_beforei
// FLOAT: call {{.*}} @{{.*}}also_beforei
// FLOAT: call {{.*}} @{{.*}}also_afterd
// FLOAT: call {{.*}} @{{.*}}also_afterl

// INT: call {{.*}} @{{.*}}also_before$ompvariant$S4$s12$Pdisable_implicit_basev
// INT: call {{.*}} @{{.*}}also_before$ompvariant$S4$s12$Pdisable_implicit_basei
// INT: call {{.*}} @{{.*}}also_before$ompvariant$S4$s12$Pdisable_implicit_basei
// INT: call {{.*}} @{{.*}}also_afterd
// INT: call {{.*}} @{{.*}}also_afterl
