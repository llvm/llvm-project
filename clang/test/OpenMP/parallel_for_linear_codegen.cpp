// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template <class T>
struct S {
  T f;
  S(T a) : f(a) {}
  S() : f() {}
  S<T> &operator=(const S<T> &);
  operator T() { return T(); }
  ~S() {}
};

volatile int g = 1212;
float f;
char cnt;

// CHECK: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK: [[S_INT_TY:%.+]] = type { i32 }
// CHECK-DAG: [[F:@.+]] = global float 0.0
// CHECK-DAG: [[CNT:@.+]] = global i8 0
template <typename T>
T tmain() {
  S<T> test;
  T *pvar = &test.f;
  T lvar = T();
#pragma omp parallel for linear(pvar, lvar)
  for (int i = 0; i < 2; ++i) {
    ++pvar, ++lvar;
  }
  return T();
}

int main() {
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // LAMBDA-LABEL: @main
  // LAMBDA: call void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i32* [[G]])
#pragma omp parallel for linear(g:5)
  for (int i = 0; i < 2; ++i) {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}})
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: [[G_START_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: store i32 0,
    // LAMBDA: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %{{.+}}
    // LAMBDA: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
    // LAMBDA: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
    // LAMBDA: [[VAL:%.+]] = load i32, i32* [[G_START_ADDR]]
    // LAMBDA: [[CNT:%.+]] = load i32, i32*
    // LAMBDA: [[MUL:%.+]] = mul nsw i32 [[CNT]], 5
    // LAMBDA: [[ADD:%.+]] = add nsw i32 [[VAL]], [[MUL]]
    // LAMBDA: store i32 [[ADD]], i32* [[G_PRIVATE_ADDR]],
    // LAMBDA: [[VAL:%.+]] = load i32, i32* [[G_PRIVATE_ADDR]],
    // LAMBDA: [[ADD:%.+]] = add nsw i32 [[VAL]], 5
    // LAMBDA: store i32 [[ADD]], i32* [[G_PRIVATE_ADDR]],
    // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: store i{{[0-9]+}}* [[G_PRIVATE_ADDR]], i{{[0-9]+}}** [[G_PRIVATE_ADDR_REF]]
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])
    // LAMBDA: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])
    g += 5;
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
      // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[G_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[G_REF]]
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // BLOCKS-LABEL: @main
  // BLOCKS: call void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i32* [[G]])
#pragma omp parallel for linear(g:5)
  for (int i = 0; i < 2; ++i) {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}})
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: [[G_START_ADDR:%.+]] = alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    // BLOCKS: store i32 0,
    // BLOCKS: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %{{.+}}
    // BLOCKS: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
    // BLOCKS: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
    // BLOCKS: [[VAL:%.+]] = load i32, i32* [[G_START_ADDR]]
    // BLOCKS: [[CNT:%.+]] = load i32, i32*
    // BLOCKS: [[MUL:%.+]] = mul nsw i32 [[CNT]], 5
    // BLOCKS: [[ADD:%.+]] = add nsw i32 [[VAL]], [[MUL]]
    // BLOCKS: store i32 [[ADD]], i32* [[G_PRIVATE_ADDR]],
    // BLOCKS: [[VAL:%.+]] = load i32, i32* [[G_PRIVATE_ADDR]],
    // BLOCKS: [[ADD:%.+]] = add nsw i32 [[VAL]], 5
    // BLOCKS: store i32 [[ADD]], i32* [[G_PRIVATE_ADDR]],
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: i{{[0-9]+}}* [[G_PRIVATE_ADDR]]
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(i8
    // BLOCKS: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])
    g += 5;
    g = 1;
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: store i{{[0-9]+}} 2, i{{[0-9]+}}*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: ret
    }();
  }
  }();
  return 0;
#else
  S<float> test;
  float *pvar = &test.f;
  long long lvar = 0;
#pragma omp parallel for linear(pvar, lvar : 3)
  for (int i = 0; i < 2; ++i) {
    pvar += 3, lvar += 3;
  }
  return tmain<int>();
#endif
}

// CHECK: define i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 2, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, float**, i64*)* [[MAIN_MICROTASK:@.+]] to void
// CHECK: = call {{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call void [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret

// CHECK: define internal void [[MAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, float** dereferenceable(8) %{{.+}}, i64* dereferenceable(8) %{{.+}})
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[PVAR_START:%.+]] = alloca float*,
// CHECK: [[LVAR_START:%.+]] = alloca i64,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[PVAR_PRIV:%.+]] = alloca float*,
// CHECK: [[LVAR_PRIV:%.+]] = alloca i64,
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// CHECK: [[PVAR_REF:%.+]] = load float**, float*** %
// CHECK: [[LVAR_REF:%.+]] = load i64*, i64** %

// Check for default initialization.
// CHECK: [[PVAR_VAL:%.+]] = load float*, float** [[PVAR_REF]],
// CHECK: store float* [[PVAR_VAL]], float** [[PVAR_START]],
// CHECK: [[LVAR_VAL:%.+]] = load i64, i64* [[LVAR_REF]],
// CHECK: store i64 [[LVAR_VAL]], i64* [[LVAR_START]],
// CHECK: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID:%.+]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
// CHECK: [[PVAR_VAL:%.+]] = load float*, float** [[PVAR_START]],
// CHECK: [[CNT:%.+]] = load i32, i32*
// CHECK: [[MUL:%.+]] = mul nsw i32 [[CNT]], 3
// CHECK: [[IDX:%.+]] = sext i32 [[MUL]] to i64
// CHECK: [[PTR:%.+]] = getelementptr inbounds float, float* [[PVAR_VAL]], i64 [[IDX]]
// CHECK: store float* [[PTR]], float** [[PVAR_PRIV]],
// CHECK: [[LVAR_VAL:%.+]] = load i64, i64* [[LVAR_START]],
// CHECK: [[CNT:%.+]] = load i32, i32*
// CHECK: [[MUL:%.+]] = mul nsw i32 [[CNT]], 3
// CHECK: [[CONV:%.+]] = sext i32 [[MUL]] to i64
// CHECK: [[VAL:%.+]] = add nsw i64 [[LVAR_VAL]], [[CONV]]
// CHECK: store i64 [[VAL]], i64* [[LVAR_PRIV]],
// CHECK: [[PVAR_VAL:%.+]] = load float*, float** [[PVAR_PRIV]]
// CHECK: [[PTR:%.+]] = getelementptr inbounds float, float* [[PVAR_VAL]], i64 3
// CHECK: store float* [[PTR]], float** [[PVAR_PRIV]],
// CHECK: [[LVAR_VAL:%.+]] = load i64, i64* [[LVAR_PRIV]],
// CHECK: [[ADD:%.+]] = add nsw i64 [[LVAR_VAL]], 3
// CHECK: store i64 [[ADD]], i64* [[LVAR_PRIV]],
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 %{{.+}})
// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 2, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i32**, i32*)* [[TMAIN_MICROTASK:@.+]] to void
// CHECK: call void [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i32** dereferenceable(8) %{{.+}}, i32* dereferenceable(4) %{{.+}})
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[PVAR_START:%.+]] = alloca i32*,
// CHECK: [[LVAR_START:%.+]] = alloca i32,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[PVAR_PRIV:%.+]] = alloca i32*,
// CHECK: [[LVAR_PRIV:%.+]] = alloca i32,
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// CHECK: [[PVAR_REF:%.+]] = load i32**, i32*** %
// CHECK: [[LVAR_REF:%.+]] = load i32*, i32** %

// Check for default initialization.
// CHECK: [[PVAR_VAL:%.+]] = load i32*, i32** [[PVAR_REF]],
// CHECK: store i32* [[PVAR_VAL]], i32** [[PVAR_START]],
// CHECK: [[LVAR_VAL:%.+]] = load i32, i32* [[LVAR_REF]],
// CHECK: store i32 [[LVAR_VAL]], i32* [[LVAR_START]],
// CHECK: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID:%.+]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
// CHECK: [[PVAR_VAL:%.+]] = load i32*, i32** [[PVAR_START]],
// CHECK: [[CNT:%.+]] = load i32, i32*
// CHECK: [[MUL:%.+]] = mul nsw i32 [[CNT]], 1
// CHECK: [[IDX:%.+]] = sext i32 [[MUL]] to i64
// CHECK: [[PTR:%.+]] = getelementptr inbounds i32, i32* [[PVAR_VAL]], i64 [[IDX]]
// CHECK: store i32* [[PTR]], i32** [[PVAR_PRIV]],
// CHECK: [[LVAR_VAL:%.+]] = load i32, i32* [[LVAR_START]],
// CHECK: [[CNT:%.+]] = load i32, i32*
// CHECK: [[MUL:%.+]] = mul nsw i32 [[CNT]], 1
// CHECK: [[VAL:%.+]] = add nsw i32 [[LVAR_VAL]], [[MUL]]
// CHECK: store i32 [[VAL]], i32* [[LVAR_PRIV]],
// CHECK: [[PVAR_VAL:%.+]] = load i32*, i32** [[PVAR_PRIV]]
// CHECK: [[PTR:%.+]] = getelementptr inbounds i32, i32* [[PVAR_VAL]], i32 1
// CHECK: store i32* [[PTR]], i32** [[PVAR_PRIV]],
// CHECK: [[LVAR_VAL:%.+]] = load i32, i32* [[LVAR_PRIV]],
// CHECK: [[ADD:%.+]] = add nsw i32 [[LVAR_VAL]], 1
// CHECK: store i32 [[ADD]], i32* [[LVAR_PRIV]],
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 %{{.+}})
// CHECK: ret void
#endif

