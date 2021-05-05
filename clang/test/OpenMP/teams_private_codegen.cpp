// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER
template <class T>
struct S {
  T f;
  S(T a) : f(a) {}
  S() : f() {}
  operator T() { return T(); }
  ~S() {}
};

volatile int g __attribute__((aligned(128))) = 1212;

struct SS {
  int a;
  int b : 4;
  int &c;
  SS(int &d) : a(0), b(0), c(d) {
#pragma omp target
#pragma omp teams private(a, b, c)
#ifdef LAMBDA
    [&]() {
      ++this->a, --b, (this)->c /= 1;
    }();
#else
    ++this->a, --b, c /= 1;
#endif
  }
};

template<typename T>
struct SST {
  T a;
  SST() : a(T()) {
#pragma omp target
#pragma omp teams private(a)
#ifdef LAMBDA
    [&]() {
      [&]() {
        ++this->a;
      }();
    }();
#else
    ++(this)->a;
#endif
  }
};

// CHECK: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// LAMBDA: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// LAMBDA: [[CAP_0_TY:%.+]] = type { [[SS_TY]]*, i{{[0-9]+}}*,
// LAMBDA: [[CAP_1_TY:%.+]] = type { i{{[0-9]+}}*, i{{[0-9]+}}* }
// CHECK: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
// CHECK: [[SST_TY:%.+]] = type { i{{[0-9]+}} }
template <typename T>
T tmain() {
  S<T> test;
  SST<T> sst;
  T t_var __attribute__((aligned(128))) = T();
  T vec[] __attribute__((aligned(128))) = {1, 2};
  S<T> s_arr[] __attribute__((aligned(128))) = {1, 2};
  S<T> var __attribute__((aligned(128))) (3);
#pragma omp target
#pragma omp teams private(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return T();
}

int main() {
  static int sivar;
  SS ss(sivar);
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] ={{.*}} global i{{[0-9]+}} 1212,
  // LAMBDA: define {{.+}} @main()
  // LAMBDA: alloca [[SS_TY]],
  // LAMBDA: alloca [[CAP_TY:%.+]],

  // LAMBDA: call{{.*}} [[ST_CONSTR_INIT:@.+]]([[SS_TY]]*
  // LAMBDA: call{{.*}} void [[OUTER_LAMBDA:@[^(]+]]([[CAP_TY]]*

  // lambda and target region in main
  // LAMBDA: define {{.+}} [[OUTER_LAMBDA]]([[CAP_TY]]* {{.+}})
  // LAMBDA: call void @[[OMP_OFFLOADING:.+]]()

  // target region in struct constructor
  // LAMBDA: define{{.*}} void [[ST_CONSTR:@.+]]([[SS_TY]]* {{[^,]*}} %this,
  // LAMBDA: call void [[OMP_OFFLOADING_1:@.+]]([[SS_TY]]

  // offloading function in struct constructor
  // LAMBDA: define{{.*}} void [[OMP_OFFLOADING_1]]([[SS_TY]]
  // LAMBDA: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[OMP_OUTLINED:@.+]] to void

  // outlined teams region in struct constructor
  // LAMBDA: define{{.*}} void [[OMP_OUTLINED]](i{{[0-9]+}}* {{.+}}, i{{[0-9]+}}* {{.+}}, [[SS_TY]]*
  // LAMBDA: [[THIS_ADDR:%.+]] = alloca [[SS_TY]]*,
  // LAMBDA: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
  // LAMBDA: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
  // LAMBDA: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
  // LAMBDA: [[THIS_REF:%.+]] = load [[SS_TY]]*, [[SS_TY]]** [[THIS_ADDR]],
  // LAMBDA: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[A_TMP_REF:%.+]],
  // LAMBDA: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[C_TMP_REF:%.+]],
  // LAMBDA: [[CAP_THIS_REF:%.+]] = getelementptr {{.+}} [[CAP_0_TY]], [[CAP_0_TY]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // LAMBDA: store [[SS_TY]]* [[THIS_REF]], [[SS_TY]]** [[CAP_THIS_REF]],
  // LAMBDA: [[CAP_A_REF:%.+]] = getelementptr {{.+}} [[CAP_0_TY]], [[CAP_0_TY]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // LAMBDA: [[A_TMP_VAL:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[A_TMP_REF]],
  // LAMBDA: store i{{[0-9]+}}* [[A_TMP_VAL]], i{{[0-9]+}}** [[CAP_A_REF]],
  // LAMBDA: [[CAP_B_REF:%.+]] = getelementptr {{.+}} [[CAP_0_TY]], [[CAP_0_TY]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // LAMBDA: store i{{[0-9]+}}* [[B_PRIV]], i{{[0-9]+}}** [[CAP_B_REF]],
  // LAMBDA: [[CAP_C_REF:%.+]] = getelementptr {{.+}} [[CAP_0_TY]], [[CAP_0_TY]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
  // LAMBDA: [[C_TMP_VAL:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[C_TMP_REF]],
  // LAMBDA: store i{{[0-9]+}}* [[C_TMP_VAL]], i{{[0-9]+}}** [[CAP_C_REF]],
  // call void [[INNER_LAMBDA_CONSTR:@.+]]([[CAP_0_TY]]*
  
  // inner lambda in struct constructor
  // define{{.*}} void [[INNER_LAMBDA_CONSTR]]([[CAP_0_TY]]*
  // LAMBDA: [[CAP_A_REF_1:%.+]] = getelementptr {{.+}} [[CAP_0_TY]], [[CAP_0_TY]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // LAMBDA: [[A_REF_FROM_CAP:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[CAP_A_REF_1]],
  // LAMBDA: [[A_VAL_FROM_CAP:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_REF_FROM_CAP]],
  // LAMBDA: [[A_INC_VAL:%.+]] = add {{.+}} i{{[0-9]+}} [[A_VAL_FROM_CAP]], 1
  // LAMBDA: store i{{[0-9]+}} [[A_INC_VAL]], i{{[0-9]+}}* [[A_REF_FROM_CAP]],

  // LAMBDA: [[CAP_B_REF_1:%.+]] = getelementptr {{.+}} [[CAP_0_TY]], [[CAP_0_TY]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // LAMBDA: [[B_REF_FROM_CAP:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[CAP_B_REF_1]],
  // LAMBDA: [[B_VAL_FROM_CAP:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_REF_FROM_CAP]],
  // LAMBDA: [[B_DEC_VAL:%.+]] = add {{.+}} i{{[0-9]+}} [[B_VAL_FROM_CAP]], -1
  // LAMBDA: store i{{[0-9]+}} [[B_DEC_VAL]], i{{[0-9]+}}* [[B_REF_FROM_CAP]],

  // LAMBDA: [[CAP_C_REF_1:%.+]] = getelementptr {{.+}} [[CAP_0_TY]], [[CAP_0_TY]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
  // LAMBDA: [[C_REF_FROM_CAP:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[CAP_C_REF_1]],
  // LAMBDA: [[C_VAL_FROM_CAP:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[C_REF_FROM_CAP]],
  // LAMBDA: [[C_DEC_VAL:%.+]] = sdiv{{.*}} i{{[0-9]+}} [[C_VAL_FROM_CAP]], 1
  // LAMBDA: store i{{[0-9]+}} [[C_DEC_VAL]], i{{[0-9]+}}* [[C_REF_FROM_CAP]],
  // ret
    
  [&]() {    
#pragma omp target
#pragma omp teams private(g, sivar)
  {
    // LAMBDA: define{{.+}} @[[OMP_OFFLOADING]]()
    // LAMBDA: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[OMP_OUTLINED_1:@.+]] to void
    
    // LAMBDA: define {{.+}} [[OMP_OUTLINED_1]](i{{[0-9]+}}* {{.+}}, i{{[0-9]+}}* {{.+}}
    // LAMBDA: [[G_LOC_OUTER:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[SIVAR_LOC_OUTER:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_LOC_OUTER]]
    // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[SIVAR_LOC_OUTER]]
    // LAMBDA: call{{.*}} void [[INNER_LAMBDA:@[^(]+]]([[CAP_1_TY]]*
    // LAMBDA: ret
    g = 1;
    sivar = 2;
    [&]() {
      // LAMBDA: define {{.+}} [[INNER_LAMBDA]]([[CAP_1_TY]]* {{.+}})
      g = 2;
      sivar = 4;
      // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}*
      // LAMBDA: store i{{[0-9]+}} 4, i{{[0-9]+}}*
    }();
  }
  }();
  return 0;
#else
  S<float> test;
  int t_var = 0;
  int vec[] = {1, 2};
  S<float> s_arr[] = {1, 2};
  S<float> var(3);
#pragma omp target
#pragma omp teams private(t_var, vec, s_arr, var, sivar)
  {
    vec[0] = t_var;
    s_arr[0] = var;
    sivar = 3;
  }
  return tmain<int>();
#endif
}

// CHECK: define{{.*}} i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* {{[^,]*}} [[TEST]])
// CHECK: call void @[[OMP_OFFLOADING:.+]]()
// CHECK: = call{{.*}} i{{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call void [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret

// target region in main function
// CHECK: define{{.+}} @[[OMP_OFFLOADING]]()
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[OMP_OUTLINED:@.+]] to void

// CHECK: define internal void [[OMP_OUTLINED]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[SIVAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: {{.+}}:
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_FLOAT_TY]]*
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* {{[^,]*}} [[S_ARR_PRIV_ITEM]])
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* {{[^,]*}} [[VAR_PRIV]])
// CHECK-DAG: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]* {{[^,]*}} [[VAR_PRIV]])
// CHECK-DAG: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]*
// CHECK: ret void

// template tmain
// CHECK: define{{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* {{[^,]*}} [[TEST]])
// CHECK: call void [[S_INT_TY_CONSTR:@.+]]([[S_INT_TY]]* {{.+}}, i{{[0-9]+}}{{.*}} 3)
// CHECK: call void [[OMP_OFFLOADING_TMAIN:@.+]]()

// target in SS constructor
// CHECK: define{{.+}} [[OMP_OFFLOADING_SS:@.+]]([[SS_TY]]*
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[OMP_OUTLINED_SS:@.+]] to void

// CHECK: define{{.*}} void [[OMP_OUTLINED_SS]](i{{[0-9]+}}* {{.+}}, i{{[0-9]+}}* {{.+}}, [[SS_TY]]*
// CHECK: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[A_REF:%.+]],
// CHECK: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[C_REF:%.+]],
// CHECK: [[A_REF_VAL:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[A_REF]]
// CHECK: [[A_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_REF_VAL]]
// CHECK: [[A_INC:%.+]] = add{{.*}} i{{[0-9]+}} [[A_VAL]], 1
// CHECK: store i{{[0-9]+}} [[A_INC]], i{{[0-9]+}}* [[A_REF_VAL]],
// CHECK: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_PRIV]]
// CHECK: [[B_DEC:%.+]] = add{{.*}} i{{[0-9]+}} [[B_VAL]], -1
// CHECK: store i{{[0-9]+}} [[B_DEC]], i{{[0-9]+}}* [[B_PRIV]],
// CHECK: [[C_REF_VAL:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[C_REF]]
// CHECK: [[C_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[C_REF_VAL]]
// CHECK: [[C_DIV:%.+]] = sdiv i{{[0-9]+}} [[C_VAL]], 1
// CHECK: store i{{[0-9]+}} [[C_DIV]], i{{[0-9]+}}* [[C_REF_VAL]],
// CHECK: ret

// target in tmain template
// CHECK: define{{.+}} [[OMP_OFFLOADING_TMAIN]]()
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[OMP_OUTLINED_TMAIN:@.+]] to void

// CHECK: define{{.*}} void [[OMP_OUTLINED_TMAIN]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}}, align 128
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}], align 128
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]], align 128
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]], align 128
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: {{.+}}:
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_INT_TY]]*
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* {{[^,]*}} [[S_ARR_PRIV_ITEM]])
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* {{[^,]*}} [[VAR_PRIV]])
// CHECK-DAG: call void [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]* {{[^,]*}} [[VAR_PRIV]])
// CHECK-DAG: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]*
// CHECK: ret

// SST constructor
// CHECK: define{{.+}} [[SST_CONST:@.+]]([[SST_TY]]* {{.+}})
// CHECK: call void [[OMP_OFFLOADING_SST:@.+]]([[SST_TY]]* {{.+}})

// target in SST constructor
// CHECK: define{{.+}} [[OMP_OFFLOADING_SST]]([[SST_TY]]* {{.+}})
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SST_TY]]*)* [[OMP_OUTLINED_SST:@.+]] to void

// CHECK: define{{.+}} [[OMP_OUTLINED_SST]](i{{[0-9]+}}* {{.+}}, i{{[0-9]+}}* noalias %{{.+}}, [[SST_TY]]* {{.+}})
// CHECK: [[A_PRIV_1:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[A_PRIV_1]], i{{[0-9]+}}** [[A_REF_1:%.+]],
// CHECK: [[A_REF_VAL_1:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[A_REF_1]]
// CHECK: [[A_VAL_1:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_REF_VAL_1]]
// CHECK: [[A_INC_1:%.+]] = add{{.*}} i{{[0-9]+}} [[A_VAL_1]], 1
// CHECK: store i{{[0-9]+}} [[A_INC_1]], i{{[0-9]+}}* [[A_REF_VAL_1]],
// CHECK: ret

#endif

