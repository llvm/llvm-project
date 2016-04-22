// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// expected-no-diagnostics
// REQUIRES: x86-registered-target
#ifndef HEADER
#define HEADER

struct SS {
  int a;
  int b : 4;
  int &c;
  SS(int &d) : a(0), b(0), c(d) {
#pragma omp parallel
#pragma omp for lastprivate(a, b, c)
    for (int i = 0; i < 2; ++i)
#ifdef LAMBDA
      [&]() {
        ++this->a, --b, (this)->c /= 1;
#pragma omp parallel
#pragma omp for lastprivate(a, b, c)
        for (int i = 0; i < 2; ++i)
          ++(this)->a, --b, this->c /= 1;
      }();
#elif defined(BLOCKS)
      ^{
        ++a;
        --this->b;
        (this)->c /= 1;
#pragma omp parallel
#pragma omp for lastprivate(a, b, c)
        for (int i = 0; i < 2; ++i)
          ++(this)->a, --b, this->c /= 1;
      }();
#else
      ++this->a, --b, c /= 1;
#endif
#pragma omp for
    for (a = 0; a < 2; ++a)
#ifdef LAMBDA
      [&]() {
        ++this->a, --b, (this)->c /= 1;
#pragma omp parallel
#pragma omp for lastprivate(b)
        for (b = 0; b < 2; ++b)
          ++(this)->a, --b, this->c /= 1;
      }();
#elif defined(BLOCKS)
      ^{
        ++a;
        --this->b;
        (this)->c /= 1;
#pragma omp parallel
#pragma omp for
        for (c = 0; c < 2; ++c)
          ++(this)->a, --b, this->c /= 1;
      }();
#else
      ++this->a, --b, c /= 1;
#endif
  }
};

template <typename T>
struct SST {
  T a;
  SST() : a(T()) {
#pragma omp parallel
#pragma omp for lastprivate(a)
    for (int i = 0; i < 2; ++i)
#ifdef LAMBDA
      [&]() {
        [&]() {
          ++this->a;
#pragma omp parallel
#pragma omp for lastprivate(a)
          for (int i = 0; i < 2; ++i)
            ++(this)->a;
        }();
      }();
#elif defined(BLOCKS)
      ^{
        ^{
          ++a;
#pragma omp parallel
#pragma omp for lastprivate(a)
          for (int i = 0; i < 2; ++i)
            ++(this)->a;
        }();
      }();
#else
      ++(this)->a;
#endif
#pragma omp for
    for (a = 0; a < 2; ++a)
#ifdef LAMBDA
      [&]() {
        ++this->a;
#pragma omp parallel
#pragma omp for
        for (a = 0; a < 2; ++(this)->a)
          ++(this)->a;
      }();
#elif defined(BLOCKS)
      ^{
        ++a;
#pragma omp parallel
#pragma omp for
        for (this->a = 0; a < 2; ++a)
          ++(this)->a;
      }();
#else
      ++(this)->a;
#endif
  }
};

template <class T>
struct S {
  T f;
  S(T a) : f(a) {}
  S() : f() {}
  S<T> &operator=(const S<T> &);
  operator T() { return T(); }
  ~S() {}
};

volatile int g __attribute__((aligned(128)))= 1212;
volatile int &g1 = g;
float f;
char cnt;

// CHECK: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// LAMBDA: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// BLOCKS: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// CHECK: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK: [[S_INT_TY:%.+]] = type { i32 }
// CHECK-DAG: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*
// CHECK-DAG: [[X:@.+]] = global double 0.0
// CHECK-DAG: [[F:@.+]] = global float 0.0
// CHECK-DAG: [[CNT:@.+]] = global i8 0
template <typename T>
T tmain() {
  S<T> test;
  SST<T> sst;
  T t_var __attribute__((aligned(128))) = T();
  T vec[] __attribute__((aligned(128))) = {1, 2};
  S<T> s_arr[] __attribute__((aligned(128))) = {1, 2};
  S<T> &var __attribute__((aligned(128))) = test;
#pragma omp parallel
#pragma omp for lastprivate(t_var, vec, s_arr, var)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
  return T();
}

namespace A {
double x;
}
namespace B {
using A::x;
}

int main() {
  static int sivar;
  SS ss(sivar);
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // LAMBDA: [[SIVAR:@.+]] = internal global i{{[0-9]+}} 0,
  // LAMBDA-LABEL: @main
  // LAMBDA: alloca [[SS_TY]],
  // LAMBDA: alloca [[CAP_TY:%.+]],
  // LAMBDA: call void [[OUTER_LAMBDA:@.+]]([[CAP_TY]]*
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i32* %{{.+}})
#pragma omp parallel
#pragma omp for lastprivate(g, g1, sivar)
  for (int i = 0; i < 2; ++i) {
    // LAMBDA: define {{.+}} @{{.+}}([[SS_TY]]*
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
    // LAMBDA: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
    // LAMBDA: store i8
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
    // LAMBDA: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[SS_MICROTASK:@.+]] to void
    // LAMBDA: call void @__kmpc_for_static_init_4(
    // LAMBDA-NOT: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
    // LAMBDA: call void {{.+}} [[SS_LAMBDA:@[^ ]+]]
    // LAMBDA: call void @__kmpc_for_static_fini(%
    // LAMBDA: ret

    // LAMBDA: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}})
    // LAMBDA: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 0
    // LAMBDA-NOT: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 1
    // LAMBDA: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 2
    // LAMBDA: call void @__kmpc_for_static_init_4(
    // LAMBDA-NOT: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]*
    // LAMBDA: call{{.*}} void
    // LAMBDA: call void @__kmpc_for_static_fini(
    // LAMBDA: br i1
    // LAMBDA: [[B_REF:%.+]] = getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 1
    // LAMBDA: store i8 %{{.+}}, i8* [[B_REF]],
    // LAMBDA: br label
    // LAMBDA: ret void

    // LAMBDA: define internal void @{{.+}}(i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}}, i32* {{.+}}, i32* {{.+}}, i32* {{.+}})
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REFA:%.+]],
    // LAMBDA: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[REFC:%.+]],
    // LAMBDA: call void @__kmpc_for_static_init_4(
    // LAMBDA: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFA]],
    // LAMBDA-NEXT: [[A_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_PRIV]],
    // LAMBDA-NEXT: [[INC:%.+]] = add nsw i{{[0-9]+}} [[A_VAL]], 1
    // LAMBDA-NEXT: store i{{[0-9]+}} [[INC]], i{{[0-9]+}}* [[A_PRIV]],
    // LAMBDA-NEXT: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_PRIV]],
    // LAMBDA-NEXT: [[DEC:%.+]] = add nsw i{{[0-9]+}} [[B_VAL]], -1
    // LAMBDA-NEXT: store i{{[0-9]+}} [[DEC]], i{{[0-9]+}}* [[B_PRIV]],
    // LAMBDA-NEXT: [[C_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFC]],
    // LAMBDA-NEXT: [[C_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[C_PRIV]],
    // LAMBDA-NEXT: [[DIV:%.+]] = sdiv i{{[0-9]+}} [[C_VAL]], 1
    // LAMBDA-NEXT: store i{{[0-9]+}} [[DIV]], i{{[0-9]+}}* [[C_PRIV]],
    // LAMBDA: call void @__kmpc_for_static_fini(
    // LAMBDA: br i1
    // LAMBDA: br label
    // LAMBDA: ret void

    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) [[SIVAR:%.+]])
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}}, align 128
    // LAMBDA: [[G1_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[SIVAR_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[SIVAR_PRIVATE_ADDR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %{{.+}},

    // LAMBDA: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %{{.+}}
    // LAMBDA: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]

    // LAMBDA: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
    // LAMBDA: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]],
    // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: store i{{[0-9]+}}* [[G_PRIVATE_ADDR]], i{{[0-9]+}}** [[G_PRIVATE_ADDR_REF]]
    // LAMBDA: [[SIVAR_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
    // LAMBDA: store i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]], i{{[0-9]+}}** [[SIVAR_PRIVATE_ADDR_REF]]
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])
    // LAMBDA: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])
    g = 1;
    g1 = 1;
    sivar = 2;
    // Check for final copying of private values back to original vars.
    // LAMBDA: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
    // LAMBDA: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
    // LAMBDA: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
    // LAMBDA: [[LAST_THEN]]
    // Actual copying.

    // original g=private_g;
    // LAMBDA: [[G_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // LAMBDA: store volatile i{{[0-9]+}} [[G_VAL]], i{{[0-9]+}}* [[G]],

    // original sivar=private_sivar;
    // LAMBDA: [[SIVAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]],
    // LAMBDA: store i{{[0-9]+}} [[SIVAR_VAL]], i{{[0-9]+}}* %{{.+}},
    // LAMBDA: br label %[[LAST_DONE]]
    // LAMBDA: [[LAST_DONE]]
    // LAMBDA: call void @__kmpc_barrier(%{{.+}}* @{{.+}}, i{{[0-9]+}} [[GTID]])
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      g1 = 2;
      sivar = 4;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
      // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[G_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[G_REF]]
      // LAMBDA: [[SIVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
      // LAMBDA: [[SIVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SIVAR_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 4, i{{[0-9]+}}* [[SIVAR_REF]]
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // BLOCKS-LABEL: @main
  // BLOCKS: call
  // BLOCKS: call void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})
#pragma omp parallel
#pragma omp for lastprivate(g, g1, sivar)
  for (int i = 0; i < 2; ++i) {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) [[SIVAR:%.+]])
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}}, align 128
    // BLOCKS: [[G1_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}}, align 4
    // BLOCKS: [[SIVAR_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    // BLOCKS: store i{{[0-9]+}}* [[SIVAR]], i{{[0-9]+}}** [[SIVAR_ADDR:%.+]],
    // BLOCKS: {{.+}} = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SIVAR_ADDR]]
    // BLOCKS: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %{{.+}}
    // BLOCKS: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
    // BLOCKS: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
    // BLOCKS: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: i{{[0-9]+}}* [[G_PRIVATE_ADDR]]
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(i8
    // BLOCKS: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])
    g = 1;
    g1 = 1;
    sivar = 2;
    // Check for final copying of private values back to original vars.
    // BLOCKS: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
    // BLOCKS: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
    // BLOCKS: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
    // BLOCKS: [[LAST_THEN]]
    // Actual copying.

    // original g=private_g;
    // BLOCKS: [[G_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // BLOCKS: store volatile i{{[0-9]+}} [[G_VAL]], i{{[0-9]+}}* [[G]],
    // BLOCKS: [[SIVAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]],
    // BLOCKS: store i{{[0-9]+}} [[SIVAR_VAL]], i{{[0-9]+}}* %{{.+}},
    // BLOCKS: br label %[[LAST_DONE]]
    // BLOCKS: [[LAST_DONE]]
    // BLOCKS: call void @__kmpc_barrier(%{{.+}}* @{{.+}}, i{{[0-9]+}} [[GTID]])
    g = 1;
    g1 = 1;
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      g1 = 1;
      sivar = 4;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: store i{{[0-9]+}} 2, i{{[0-9]+}}*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
      // BLOCKS: store i{{[0-9]+}} 4, i{{[0-9]+}}*
      // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
      // BLOCKS: ret
    }();
  }
  }();
  return 0;
// BLOCKS: define {{.+}} @{{.+}}([[SS_TY]]*
// BLOCKS: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
// BLOCKS: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
// BLOCKS: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
// BLOCKS: store i8
// BLOCKS: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
// BLOCKS: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[SS_MICROTASK:@.+]] to void
// BLOCKS: call void @__kmpc_for_static_init_4(
// BLOCKS-NOT: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
// BLOCKS: call void
// BLOCKS: call void @__kmpc_for_static_fini(%
// BLOCKS: ret

// BLOCKS: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}})
// BLOCKS: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 0
// BLOCKS-NOT: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 1
// BLOCKS: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 2
// BLOCKS: call void @__kmpc_for_static_init_4(
// BLOCKS-NOT: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]*
// BLOCKS: call{{.*}} void
// BLOCKS: call void @__kmpc_for_static_fini(
// BLOCKS: br i1
// BLOCKS: [[B_REF:%.+]] = getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 1
// BLOCKS: store i8 %{{.+}}, i8* [[B_REF]],
// BLOCKS: br label
// BLOCKS: ret void

// BLOCKS: define internal void @{{.+}}(i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}}, i32* {{.+}}, i32* {{.+}}, i32* {{.+}})
// BLOCKS: alloca i{{[0-9]+}},
// BLOCKS: alloca i{{[0-9]+}},
// BLOCKS: alloca i{{[0-9]+}},
// BLOCKS: alloca i{{[0-9]+}},
// BLOCKS: alloca i{{[0-9]+}},
// BLOCKS: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
// BLOCKS: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
// BLOCKS: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
// BLOCKS: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REFA:%.+]],
// BLOCKS: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[REFC:%.+]],
// BLOCKS: call void @__kmpc_for_static_init_4(
// BLOCKS: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFA]],
// BLOCKS-NEXT: [[A_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_PRIV]],
// BLOCKS-NEXT: [[INC:%.+]] = add nsw i{{[0-9]+}} [[A_VAL]], 1
// BLOCKS-NEXT: store i{{[0-9]+}} [[INC]], i{{[0-9]+}}* [[A_PRIV]],
// BLOCKS-NEXT: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_PRIV]],
// BLOCKS-NEXT: [[DEC:%.+]] = add nsw i{{[0-9]+}} [[B_VAL]], -1
// BLOCKS-NEXT: store i{{[0-9]+}} [[DEC]], i{{[0-9]+}}* [[B_PRIV]],
// BLOCKS-NEXT: [[C_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFC]],
// BLOCKS-NEXT: [[C_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[C_PRIV]],
// BLOCKS-NEXT: [[DIV:%.+]] = sdiv i{{[0-9]+}} [[C_VAL]], 1
// BLOCKS-NEXT: store i{{[0-9]+}} [[DIV]], i{{[0-9]+}}* [[C_PRIV]],
// BLOCKS: call void @__kmpc_for_static_fini(
// BLOCKS: br i1
// BLOCKS: br label
// BLOCKS: ret void
#else
  S<float> test;
  int t_var = 0;
  int vec[] = {1, 2};
  S<float> s_arr[] = {1, 2};
  S<float> var(3);
#pragma omp parallel
#pragma omp for lastprivate(t_var, vec, s_arr, var, sivar)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
    sivar += i;
  }
#pragma omp parallel
#pragma omp for lastprivate(A::x, B::x) firstprivate(f) lastprivate(f)
  for (int i = 0; i < 2; ++i) {
    A::x++;
  }
#pragma omp parallel
#pragma omp for firstprivate(f) lastprivate(f)
  for (int i = 0; i < 2; ++i) {
    A::x++;
  }
#pragma omp parallel
#pragma omp for lastprivate(cnt)
  for (cnt = 0; cnt < 2; ++cnt) {
    A::x++;
  }
  return tmain<int>();
#endif
}

// CHECK: define i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 5, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i32*, [2 x i32]*, [2 x [[S_FLOAT_TY]]]*, [[S_FLOAT_TY]]*, i32*)* [[MAIN_MICROTASK:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[MAIN_MICROTASK1:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[MAIN_MICROTASK2:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[MAIN_MICROTASK3:@.+]] to void
// CHECK: = call {{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call void [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret

// CHECK: define internal void [[MAIN_MICROTASK]](i32* noalias [[GTID_ADDR:%.+]], i32* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}}, [2 x i32]* dereferenceable(8) %{{.+}}, [2 x [[S_FLOAT_TY]]]* dereferenceable(8) %{{.+}}, [[S_FLOAT_TY]]* dereferenceable(4) %{{.+}})
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[SIVAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// CHECK: [[T_VAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %
// CHECK: [[VEC_REF:%.+]] = load [2 x i32]*, [2 x i32]** %
// CHECK: [[S_ARR_REF:%.+]] = load [2 x [[S_FLOAT_TY]]]*, [2 x [[S_FLOAT_TY]]]** %
// CHECK: [[VAR_REF:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** %

// Check for default initialization.
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_FLOAT_TY]]*
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* [[S_ARR_PRIV_ITEM]])
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 %{{.+}}, i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
// <Skip loop body>
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 %{{.+}})

// Check for final copying of private values back to original vars.
// CHECK: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
// CHECK: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
// CHECK: [[LAST_THEN]]
// Actual copying.

// original t_var=private_t_var;
// CHECK: [[T_VAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_PRIV]],
// CHECK: store i{{[0-9]+}} [[T_VAR_VAL]], i{{[0-9]+}}* [[T_VAR_REF]],

// original vec[]=private_vec[];
// CHECK: [[VEC_DEST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_REF]] to i8*
// CHECK: [[VEC_SRC:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[VEC_DEST]], i8* [[VEC_SRC]],

// original s_arr[]=private_s_arr[];
// CHECK: [[S_ARR_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_FLOAT_TY]]], [2 x [[S_FLOAT_TY]]]* [[S_ARR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = bitcast [2 x [[S_FLOAT_TY]]]* [[S_ARR_PRIV]] to [[S_FLOAT_TY]]*
// CHECK: [[S_ARR_END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[S_ARR_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[S_ARR_BEGIN]], [[S_ARR_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN:@.+]]([[S_FLOAT_TY]]* {{.+}}, [[S_FLOAT_TY]]* {{.+}})
// CHECK: br i1 {{.+}}, label %[[S_ARR_BODY_DONE]], label %[[S_ARR_BODY]]
// CHECK: [[S_ARR_BODY_DONE]]

// original var=private_var;
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN:@.+]]([[S_FLOAT_TY]]* [[VAR_REF]], [[S_FLOAT_TY]]* {{.*}} [[VAR_PRIV]])
// CHECK: [[SIVAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIVAR_PRIV]],
// CHECK: br label %[[LAST_DONE]]
// CHECK: [[LAST_DONE]]
// CHECK-DAG: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]*
// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void

//
// CHECK: define internal void [[MAIN_MICROTASK1]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[F_PRIV:%.+]] = alloca float,
// CHECK-NOT: alloca float
// CHECK: [[X_PRIV:%.+]] = alloca double,
// CHECK-NOT: alloca float
// CHECK-NOT: alloca double

// Check for default initialization.
// CHECK-NOT: [[X_PRIV]]
// CHECK: [[F_VAL:%.+]] = load float, float* [[F]],
// CHECK: store float [[F_VAL]], float* [[F_PRIV]],
// CHECK-NOT: [[X_PRIV]]

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
// <Skip loop body>
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])

// Check for final copying of private values back to original vars.
// CHECK: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
// CHECK: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
// CHECK: [[LAST_THEN]]
// Actual copying.

// original x=private_x;
// CHECK: [[X_VAL:%.+]] = load double, double* [[X_PRIV]],
// CHECK: store double [[X_VAL]], double* [[X]],

// original f=private_f;
// CHECK: [[F_VAL:%.+]] = load float, float* [[F_PRIV]],
// CHECK: store float [[F_VAL]], float* [[F]],

// CHECK-NEXT: br label %[[LAST_DONE]]
// CHECK: [[LAST_DONE]]

// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK2]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[F_PRIV:%.+]] = alloca float,
// CHECK-NOT: alloca float

// Check for default initialization.
// CHECK: [[F_VAL:%.+]] = load float, float* [[F]],
// CHECK: store float [[F_VAL]], float* [[F_PRIV]],

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
// <Skip loop body>
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])

// Check for final copying of private values back to original vars.
// CHECK: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
// CHECK: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
// CHECK: [[LAST_THEN]]
// Actual copying.

// original f=private_f;
// CHECK: [[F_VAL:%.+]] = load float, float* [[F_PRIV]],
// CHECK: store float [[F_VAL]], float* [[F]],

// CHECK-NEXT: br label %[[LAST_DONE]]
// CHECK: [[LAST_DONE]]

// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK3]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[CNT_PRIV:%.+]] = alloca i8,

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 1)
// UB = min(UB, GlobalUB)
// CHECK-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[UBCMP:%.+]] = icmp sgt i32 [[UB]], 1
// CHECK-NEXT: br i1 [[UBCMP]], label [[UB_TRUE:%[^,]+]], label [[UB_FALSE:%[^,]+]]
// CHECK: [[UBRESULT:%.+]] = phi i32 [ 1, [[UB_TRUE]] ], [ [[UBVAL:%[^,]+]], [[UB_FALSE]] ]
// CHECK-NEXT: store i32 [[UBRESULT]], i32* [[OMP_UB]]
// CHECK-NEXT: [[LB:%.+]] = load i32, i32* [[OMP_LB]]
// CHECK-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]
// <Skip loop body>
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])

// Check for final copying of private values back to original vars.
// CHECK: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
// CHECK: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
// CHECK: [[LAST_THEN]]

// Calculate private cnt value.
// CHECK: store i8 2, i8* [[CNT_PRIV]]
// original cnt=private_cnt;
// CHECK: [[CNT_VAL:%.+]] = load i8, i8* [[CNT_PRIV]],
// CHECK: store i8 [[CNT_VAL]], i8* [[CNT]],

// CHECK-NEXT: br label %[[LAST_DONE]]
// CHECK: [[LAST_DONE]]

// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i32*, [2 x i32]*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]*)* [[TMAIN_MICROTASK:@.+]] to void
// CHECK: call void [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret

// CHECK: define {{.+}} @{{.+}}([[SS_TY]]*
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
// CHECK: store i8
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[SS_MICROTASK:@.+]] to void
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-NOT: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
// CHECK: call void @__kmpc_for_static_fini(%
// CHECK: ret

// CHECK: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}})
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REFA:%.+]],
// CHECK: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[REFC:%.+]],
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFA]],
// CHECK-NEXT: [[A_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_PRIV]],
// CHECK-NEXT: [[INC:%.+]] = add nsw i{{[0-9]+}} [[A_VAL]], 1
// CHECK-NEXT: store i{{[0-9]+}} [[INC]], i{{[0-9]+}}* [[A_PRIV]],
// CHECK-NEXT: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_PRIV]],
// CHECK-NEXT: [[DEC:%.+]] = add nsw i{{[0-9]+}} [[B_VAL]], -1
// CHECK-NEXT: store i{{[0-9]+}} [[DEC]], i{{[0-9]+}}* [[B_PRIV]],
// CHECK-NEXT: [[C_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFC]],
// CHECK-NEXT: [[C_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[C_PRIV]],
// CHECK-NEXT: [[DIV:%.+]] = sdiv i{{[0-9]+}} [[C_VAL]], 1
// CHECK-NEXT: store i{{[0-9]+}} [[DIV]], i{{[0-9]+}}* [[C_PRIV]],
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: br i1
// CHECK: [[B_REF:%.+]] = getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 1
// CHECK: store i8 %{{.+}}, i8* [[B_REF]],
// CHECK: br label
// CHECK: ret void

// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}}, [2 x i32]* dereferenceable(8) %{{.+}}, [2 x [[S_INT_TY]]]* dereferenceable(8) %{{.+}}, [[S_INT_TY]]* dereferenceable(4) %{{.+}})
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}}, align 128
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}], align 128
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]], align 128
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]], align 128
// CHECK: [[VAR_PRIV_REF:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// CHECK: [[T_VAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %
// CHECK: [[VEC_REF:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** %
// CHECK: [[S_ARR_REF:%.+]] = load [2 x [[S_INT_TY]]]*, [2 x [[S_INT_TY]]]** %

// Check for default initialization.
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_INT_TY]]*
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[S_ARR_PRIV_ITEM]])
// CHECK: [[VAR_REF:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** %
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK: store [[S_INT_TY]]* [[VAR_PRIV]], [[S_INT_TY]]** [[VAR_PRIV_REF]]
// CHECK: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 %{{.+}}, i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
// <Skip loop body>
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 %{{.+}})

// Check for final copying of private values back to original vars.
// CHECK: [[IS_LAST_VAL:%.+]] = load i32, i32* [[IS_LAST_ADDR]],
// CHECK: [[IS_LAST_ITER:%.+]] = icmp ne i32 [[IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_ITER:%.+]], label %[[LAST_THEN:.+]], label %[[LAST_DONE:.+]]
// CHECK: [[LAST_THEN]]
// Actual copying.

// original t_var=private_t_var;
// CHECK: [[T_VAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_PRIV]],
// CHECK: store i{{[0-9]+}} [[T_VAR_VAL]], i{{[0-9]+}}* [[T_VAR_REF]],

// original vec[]=private_vec[];
// CHECK: [[VEC_DEST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_REF]] to i8*
// CHECK: [[VEC_SRC:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[VEC_DEST]], i8* [[VEC_SRC]],

// original s_arr[]=private_s_arr[];
// CHECK: [[S_ARR_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[S_ARR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = bitcast [2 x [[S_INT_TY]]]* [[S_ARR_PRIV]] to [[S_INT_TY]]*
// CHECK: [[S_ARR_END:%.+]] = getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_INT_TY]]* [[S_ARR_BEGIN]], [[S_ARR_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN:@.+]]([[S_INT_TY]]* {{.+}}, [[S_INT_TY]]* {{.+}})
// CHECK: br i1 {{.+}}, label %[[S_ARR_BODY_DONE]], label %[[S_ARR_BODY]]
// CHECK: [[S_ARR_BODY_DONE]]

// original var=private_var;
// CHECK: [[VAR_PRIV1:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** [[VAR_PRIV_REF]],
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN:@.+]]([[S_INT_TY]]* [[VAR_REF]], [[S_INT_TY]]* {{.*}} [[VAR_PRIV1]])
// CHECK: br label %[[LAST_DONE]]
// CHECK: [[LAST_DONE]]
// CHECK-DAG: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]*
// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void
#endif

