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

enum omp_allocator_handle_t {
  omp_null_allocator = 0,
  omp_default_mem_alloc = 1,
  omp_large_cap_mem_alloc = 2,
  omp_const_mem_alloc = 3,
  omp_high_bw_mem_alloc = 4,
  omp_low_lat_mem_alloc = 5,
  omp_cgroup_mem_alloc = 6,
  omp_pteam_mem_alloc = 7,
  omp_thread_mem_alloc = 8,
  KMP_ALLOCATOR_MAX_HANDLE = __UINTPTR_MAX__
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

volatile int g = 1212;
volatile int &g1 = g;
float f;
char cnt;

struct SS {
  int a;
  int b : 4;
  int &c;
  SS(int &d) : a(0), b(0), c(d) {
#pragma omp parallel
#pragma omp for linear(a, b, c)
    for (int i = 0; i < 2; ++i)
#ifdef LAMBDA
      [&]() {
        ++this->a, --b, (this)->c /= 1;
#pragma omp parallel
#pragma omp for linear(a, b) linear(ref(c))
        for (int i = 0; i < 2; ++i)
          ++(this)->a, --b, this->c /= 1;
      }();
#elif defined(BLOCKS)
      ^{
        ++a;
        --this->b;
        (this)->c /= 1;
#pragma omp parallel
#pragma omp for linear(a, b) linear(uval(c))
        for (int i = 0; i < 2; ++i)
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
#pragma omp for linear(a)
    for (int i = 0; i < 2; ++i)
#ifdef LAMBDA
      [&]() {
        [&]() {
          ++this->a;
#pragma omp parallel
#pragma omp for linear(a)
          for (int i = 0; i < 2; ++i)
            ++(this)->a;
        }();
      }();
#elif defined(BLOCKS)
      ^{
        ^{
          ++a;
#pragma omp parallel
#pragma omp for linear(a)
          for (int i = 0; i < 2; ++i)
            ++(this)->a;
        }();
      }();
#else
      ++(this)->a;
#endif
  }
};

// CHECK: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// LAMBDA: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// BLOCKS: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// CHECK: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK: [[S_INT_TY:%.+]] = type { i32 }
// CHECK-DAG: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*
// CHECK-DAG: [[F:@.+]] = global float 0.0
// CHECK-DAG: [[CNT:@.+]] = global i8 0
template <typename T>
T tmain() {
  S<T> test;
  SST<T> sst;
  T *pvar = &test.f;
  T &lvar = test.f;
#pragma omp parallel
#pragma omp for linear(pvar, lvar)
  for (int i = 0; i < 2; ++i) {
    ++pvar, ++lvar;
  }
  return T();
}

int main() {
  static int sivar;
  SS ss(sivar);
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // LAMBDA-LABEL: @main
  // LAMBDA: alloca [[SS_TY]],
  // LAMBDA: alloca [[CAP_TY:%.+]],
  // LAMBDA: call void [[OUTER_LAMBDA:@.+]]([[CAP_TY]]*
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})
#pragma omp parallel
#pragma omp for linear(g, g1:5)
  for (int i = 0; i < 2; ++i) {
    // LAMBDA: define {{.+}} @{{.+}}([[SS_TY]]*
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
    // LAMBDA: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
    // LAMBDA: store i8
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
    // LAMBDA: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[SS_MICROTASK:@.+]] to void
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

    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: [[G_START_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
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
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](%{{.+}}* {{[^,]*}} [[ARG]])
    // LAMBDA: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])
    g += 5;
    g1 += 5;
    // LAMBDA: call void @__kmpc_barrier(%{{.+}}* @{{.+}}, i{{[0-9]+}} [[GTID]])
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* {{[^,]*}} [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      g1 = 2;
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
  // BLOCKS: call
  // BLOCKS: call void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})
#pragma omp parallel
#pragma omp for linear(g, g1:5)
  for (int i = 0; i < 2; ++i) {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: [[G_START_ADDR:%.+]] = alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: alloca i{{[0-9]+}},
    // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
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
    g1 += 5;
    // BLOCKS: call void @__kmpc_barrier(%{{.+}}* @{{.+}}, i{{[0-9]+}} [[GTID]])
    g = 1;
    g1 = 5;
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      g1 = 2;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: store i{{[0-9]+}} 2, i{{[0-9]+}}*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
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
  float *pvar = &test.f;
  long long lvar = 0;
#pragma omp parallel
#pragma omp for linear(pvar, lvar : 3) allocate(omp_low_lat_mem_alloc: lvar)
  for (int i = 0; i < 2; ++i) {
    pvar += 3, lvar += 3;
  }
  return tmain<int>();
#endif
}

// CHECK: define i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* {{[^,]*}} [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 2, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, float**, i64*)* [[MAIN_MICROTASK:@.+]] to void
// CHECK: = call {{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call void [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret

// CHECK: define internal void [[MAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, i64* nonnull align 8 dereferenceable(8) %{{.+}})
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[PVAR_START:%.+]] = alloca float*,
// CHECK: [[LVAR_START:%.+]] = alloca i64,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[PVAR_PRIV:%.+]] = alloca float*,
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// Check for default initialization.
// CHECK: [[PVAR_REF:%.+]] = load float**, float*** %
// CHECK: [[LVAR_REF:%.+]] = load i64*, i64** %
// CHECK: [[PVAR_VAL:%.+]] = load float*, float** [[PVAR_REF]],
// CHECK: store float* [[PVAR_VAL]], float** [[PVAR_START]],
// CHECK: [[LVAR_VAL:%.+]] = load i64, i64* [[LVAR_REF]],
// CHECK: store i64 [[LVAR_VAL]], i64* [[LVAR_START]],
// CHECK: [[LVAR_VOID_PTR:%.+]] = call i8* @__kmpc_alloc(i32 [[GTID:%.+]], i64 8, i8* inttoptr (i64 5 to i8*))
// CHECK: [[LVAR_PRIV:%.+]] = bitcast i8* [[LVAR_VOID_PTR]] to i64*
// CHECK: call {{.+}} @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_ADDR:%.+]], i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32 1, i32 1)
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
// CHECK: [[LVAR_VOID_PTR:%.+]] = bitcast i64* [[LVAR_PRIV]] to i8*
// CHECK: call void @__kmpc_free(i32 [[GTID]], i8* [[LVAR_VOID_PTR]], i8* inttoptr (i64 5 to i8*))
// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* {{[^,]*}} [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 2, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i32**, i32*)* [[TMAIN_MICROTASK:@.+]] to void
// CHECK: call void [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret

// CHECK: define {{.+}} @{{.+}}([[SS_TY]]*
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
// CHECK: store i8
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[SS_MICROTASK:@.+]] to void
// CHECK: ret

// CHECK: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}})
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: call void @__kmpc_barrier(
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

// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i32** nonnull align 8 dereferenceable(8) %{{.+}}, i32* nonnull align 4 dereferenceable(4) %{{.+}})
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
// CHECK: [[LVAR_PRIV_REF:%.+]] = alloca i32*,
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// Check for default initialization.
// CHECK: [[PVAR_REF:%.+]] = load i32**, i32*** %
// CHECK: [[PVAR_VAL:%.+]] = load i32*, i32** [[PVAR_REF]],
// CHECK: store i32* [[PVAR_VAL]], i32** [[PVAR_START]],
// CHECK: [[LVAR_REF:%.+]] = load i32*, i32** %
// CHECK: [[LVAR_VAL:%.+]] = load i32, i32* [[LVAR_REF]],
// CHECK: store i32 [[LVAR_VAL]], i32* [[LVAR_START]],
// CHECK: store i32* [[LVAR_PRIV]], i32** [[LVAR_PRIV_REF]],

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
// CHECK: [[LVAR_PRIV:%.+]] = load i32*, i32** [[LVAR_PRIV_REF]],
// CHECK: [[LVAR_VAL:%.+]] = load i32, i32* [[LVAR_PRIV]],
// CHECK: [[ADD:%.+]] = add nsw i32 [[LVAR_VAL]], 1
// CHECK: store i32 [[ADD]], i32* [[LVAR_PRIV]],
// CHECK: call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 %{{.+}})
// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])
// CHECK: ret void
#endif

