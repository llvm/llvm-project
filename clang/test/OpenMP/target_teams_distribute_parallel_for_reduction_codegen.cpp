// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++  -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64

// RUN: %clang_cc1 -DCHECK -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY 
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY 
// RUN: %clang_cc1 -DCHECK -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++  -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY 
// SIMD-ONLY-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template <typename T>
T tmain() {
  T t_var = T();
  T vec[] = {1, 2};
#pragma omp target teams distribute parallel for reduction(+: t_var)
  for (int i = 0; i < 2; ++i) {
    t_var += (T) i;
  }
  return T();
}

int main() {
  static int sivar;
#ifdef LAMBDA
  // LAMBDA: [[RED_VAR:@.+]] = common global [8 x {{.+}}] zeroinitializer

  // LAMBDA-LABEL: @main
  // LAMBDA: call void [[OUTER_LAMBDA:@.+]](
  [&]() {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
    // LAMBDA: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 1, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i8** null, i8** null, i32 0, i32 0)
    // LAMBDA: call void @[[LOFFL1:.+]](
    // LAMBDA:  ret
#pragma omp target teams distribute parallel for reduction(+: sivar)
  for (int i = 0; i < 2; ++i) {
    // LAMBDA: define{{.*}} internal{{.*}} void @[[LOFFL1]](i{{64|32}}*{{.*}} [[SIVAR_ARG:%.+]])
    // LAMBDA: [[SIVAR_ADDR:%.+]] = alloca i{{.+}},
    // LAMBDA: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
    // LAMBDA: [[SIVAR_PAR:%.+]] = load{{.+}}, {{.+}} [[SIVAR_ADDR]],
    // LAMBDA: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[LOUTL1:.+]] to {{.+}}, {{.+}} [[SIVAR_PAR]])
    // LAMBDA: ret void

    // LAMBDA: define internal void @[[LOUTL1]]({{.+}}, {{.+}}, {{.+}} [[SIVAR_ARG:%.+]])
    // Skip global and bound tid vars
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: [[SIVAR_ADDR:%.+]] = alloca i{{.+}},
    // LAMBDA: [[SIVAR_PRIV:%.+]] = alloca i{{.+}},
    // LAMBDA: [[RED_LIST:%.+]] = alloca [1 x {{.+}}],
    // LAMBDA: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
    // LAMBDA: [[SIVAR_REF:%.+]] = load{{.+}}, {{.+}} [[SIVAR_ADDR]]
    // LAMBDA: store{{.+}} 0, {{.+}} [[SIVAR_PRIV]],

    // LAMBDA: call void @__kmpc_for_static_init_4(
    // LAMBDA: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[LPAR_OUTL:.+]] to
    // LAMBDA: call void @__kmpc_for_static_fini(
    // LAMBDA: [[RED_LIST_GEP:%.+]] = getelementptr{{.+}} [[RED_LIST]],
    // LAMBDA: [[SIVAR_PRIV_CAST:%.+]] = bitcast{{.+}} [[SIVAR_PRIV]] to
    // LAMBDA: store{{.+}} [[SIVAR_PRIV_CAST]], {{.+}} [[RED_LIST_GEP]],
    // LAMBDA: [[RED_LIST_BCAST:%.+]] = bitcast{{.+}} [[RED_LIST]] to
    // LAMBDA: [[K_RED_RET:%.+]] = call{{.+}} @__kmpc_reduce_nowait({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[RED_LIST_BCAST]], {{.+}} [[RED_FUN:@.+]], {{.+}} [[RED_VAR]])
    // LAMBDA: switch{{.+}} [[K_RED_RET]], label{{.+}} [
    // LAMBDA: {{.+}}, label %[[CASE1:.+]]
    // LAMBDA: {{.+}}, label %[[CASE2:.+]]
    // LAMBDA: ]
    // LAMBDA: [[CASE1]]:
    // LAMBDA-DAG: [[SIVAR_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_REF]],
    // LAMBDA-DAG: [[SIVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_PRIV]],
    // LAMBDA-DAG: [[SIVAR_INC:%.+]] = add{{.+}} [[SIVAR_VAL]], [[SIVAR_PRIV_VAL]]
    // LAMBDA: store{{.+}} [[SIVAR_INC]], {{.+}} [[SIVAR_REF]],
    // LAMBDA: call void @__kmpc_end_reduce_nowait({{.+}}, {{.+}}, {{.+}} [[RED_VAR]])
    // LAMBDA: br
    // LAMBDA: [[CASE2]]:
    // LAMBDA-DAG: [[SIVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_PRIV]],
    // LAMBDA-DAG: [[ATOMIC_RES:%.+]] = atomicrmw add{{.+}} [[SIVAR_REF]], {{.+}} [[SIVAR_PRIV_VAL]] monotonic, align {{.+}}
    // LAMBDA: br

    // LAMBDA: define internal void @[[LPAR_OUTL]]({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[SIVAR_ARG:%.+]])

    // Skip global and bound tid vars, and prev lb and ub vars
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: [[SIVAR_ADDR:%.+]] = alloca i{{.+}},
    // skip loop vars
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: [[SIVAR_PRIV:%.+]] = alloca i{{.+}},
    // LAMBDA: [[RED_LIST:%.+]] = alloca [1 x {{.+}}],
    // LAMBDA: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
    // LAMBDA: [[SIVAR_REF:%.+]] = load {{.+}} [[SIVAR_ADDR]]
    // LAMBDA: store{{.+}} 0, {{.+}} [[SIVAR_PRIV]],

    // LAMBDA: call void @__kmpc_for_static_init_4(
     // LAMBDA: store{{.+}}, {{.+}} [[SIVAR_PRIV]],
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](
    // LAMBDA: call void @__kmpc_for_static_fini(
    // LAMBDA: [[RED_LIST_GEP:%.+]] = getelementptr{{.+}} [[RED_LIST]],
    // LAMBDA: [[SIVAR_PRIV_CAST:%.+]] = bitcast{{.+}} [[SIVAR_PRIV]] to
    // LAMBDA: store{{.+}} [[SIVAR_PRIV_CAST]], {{.+}} [[RED_LIST_GEP]],
    // LAMBDA: [[RED_LIST_BCAST:%.+]] = bitcast{{.+}} [[RED_LIST]] to
    // LAMBDA: [[K_RED_RET:%.+]] = call{{.+}} @__kmpc_reduce_nowait({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[RED_LIST_BCAST]], {{.+}} [[RED_FUN:@.+]], {{.+}} [[RED_VAR]])
    // LAMBDA: switch{{.+}} [[K_RED_RET]], label{{.+}} [
    // LAMBDA: {{.+}}, label %[[CASE1:.+]]
    // LAMBDA: {{.+}}, label %[[CASE2:.+]]
    // LAMBDA: ]
    // LAMBDA: [[CASE1]]:
    // LAMBDA-64-DAG: [[SIVAR_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_REF]],
    // LAMBDA-32-DAG: [[SIVAR_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_ADDR]],
    // LAMBDA-DAG: [[SIVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_PRIV]],
    // LAMBDA-DAG: [[SIVAR_INC:%.+]] = add{{.+}} [[SIVAR_VAL]], [[SIVAR_PRIV_VAL]]
    // LAMBDA: store{{.+}} [[SIVAR_INC]], {{.+}} [[SIVAR_REF]],
    // LAMBDA: call void @__kmpc_end_reduce_nowait({{.+}}, {{.+}}, {{.+}} [[RED_VAR]])
    // LAMBDA: br
    // LAMBDA: [[CASE2]]:
    // LAMBDA-DAG: [[SIVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_PRIV]],
    // LAMBDA-DAG: [[ATOMIC_RES:%.+]] = atomicrmw add{{.+}} [[SIVAR_REF]], {{.+}} [[SIVAR_PRIV_VAL]] monotonic, align {{.+}}
    // LAMBDA: br

    sivar += i;

    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]]({{.+}} [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],

      sivar += 4;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]

      // LAMBDA: [[SIVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[SIVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SIVAR_PTR_REF]]
      // LAMBDA: [[SIVAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIVAR_REF]]
      // LAMBDA: [[SIVAR_INC:%.+]] = add{{.+}} [[SIVAR_VAL]], 4
      // LAMBDA: store i{{[0-9]+}} [[SIVAR_INC]], i{{[0-9]+}}* [[SIVAR_REF]]
    }();
  }
  }();
  return 0;
#else
#pragma omp target teams distribute parallel for reduction(+: sivar)
  for (int i = 0; i < 2; ++i) {
    sivar += i;
  }
  return tmain<int>();
#endif
}

// CHECK: [[RED_VAR:@.+]] = common global [8 x {{.+}}] zeroinitializer

// CHECK: define {{.*}}i{{[0-9]+}} @main()
// CHECK: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 1, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i8** null, i8** null, i32 0, i32 0)
// CHECK: call void @[[OFFL1:.+]](i{{64|32}}* @{{.+}})
// CHECK: {{%.+}} = call{{.*}} i32 @[[TMAIN_INT:.+]]()
// CHECK:  ret

// CHECK: define{{.*}} void @[[OFFL1]](i{{64|32}}*{{.*}} [[SIVAR_ARG:%.+]])
// CHECK: [[SIVAR_ADDR:%.+]] = alloca i{{.+}},
// CHECK: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
// CHECK: [[SIVAR_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_ADDR]]
// CHECK: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTL1:.+]] to {{.+}}, {{.+}} [[SIVAR_VAL]])
// CHECK: ret void

// CHECK: define internal void @[[OUTL1]]({{.+}}, {{.+}}, {{.+}} [[SIVAR_ARG:%.+]])
// Skip global and bound tid vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: [[SIVAR_ADDR:%.+]] = alloca i{{.+}},
// CHECK: [[SIVAR_PRIV:%.+]] = alloca i{{.+}},
// CHECK: [[RED_LIST:%.+]] = alloca [1 x {{.+}}],
// CHECK: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
// CHECK: [[SIVAR_REF:%.+]] = load{{.+}} [[SIVAR_ADDR]],
// CHECK: store{{.+}} 0, {{.+}} [[SIVAR_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL:.+]] to
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: [[RED_LIST_GEP:%.+]] = getelementptr{{.+}} [[RED_LIST]],
// CHECK: [[SIVAR_PRIV_CAST:%.+]] = bitcast{{.+}} [[SIVAR_PRIV]] to
// CHECK: store{{.+}} [[SIVAR_PRIV_CAST]], {{.+}} [[RED_LIST_GEP]],
// CHECK: [[RED_LIST_BCAST:%.+]] = bitcast{{.+}} [[RED_LIST]] to
// CHECK: [[K_RED_RET:%.+]] = call{{.+}} @__kmpc_reduce_nowait({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[RED_LIST_BCAST]], {{.+}} [[RED_FUN:@.+]], {{.+}} [[RED_VAR]])
// CHECK: switch{{.+}} [[K_RED_RET]], label{{.+}} [
// CHECK: {{.+}}, label %[[CASE1:.+]]
// CHECK: {{.+}}, label %[[CASE2:.+]]
// CHECK: ]
// CHECK: [[CASE1]]:
// CHECK-DAG: [[SIVAR_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_REF]],
// CHECK-DAG: [[SIVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_PRIV]],
// CHECK-DAG: [[SIVAR_INC:%.+]] = add{{.+}} [[SIVAR_VAL]], [[SIVAR_PRIV_VAL]]
// CHECK: store{{.+}} [[SIVAR_INC]], {{.+}} [[SIVAR_REF]],
// CHECK: call void @__kmpc_end_reduce_nowait({{.+}}, {{.+}}, {{.+}} [[RED_VAR]])
// CHECK: br
// CHECK: [[CASE2]]:
// CHECK-DAG: [[SIVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_PRIV]],
// CHECK-DAG: [[ATOMIC_RES:%.+]] = atomicrmw add{{.+}} [[SIVAR_REF]], {{.+}} [[SIVAR_PRIV_VAL]] monotonic, align {{.+}}
// CHECK: br

// CHECK: define internal void @[[PAR_OUTL]]({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[SIVAR_ARG:%.+]])
// Skip global and bound tid vars, and prev lb and ub
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[SIVAR_ADDR:%.+]] = alloca i{{.+}},
// skip loop vars
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: [[SIVAR_PRIV:%.+]] = alloca i{{.+}},
// CHECK: [[RED_LIST:%.+]] = alloca [1 x {{.+}}],
// CHECK: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
// CHECK-64: [[SIVAR_REF:%.+]] = load {{.+}} [[SIVAR_ADDR]],
// CHECK: store{{.+}} 0, {{.+}} [[SIVAR_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: store{{.+}}, {{.+}} [[SIVAR_PRIV]],
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: [[RED_LIST_GEP:%.+]] = getelementptr{{.+}} [[RED_LIST]],
// CHECK: [[SIVAR_PRIV_CAST:%.+]] = bitcast{{.+}} [[SIVAR_PRIV]] to
// CHECK: store{{.+}} [[SIVAR_PRIV_CAST]], {{.+}} [[RED_LIST_GEP]],
// CHECK: [[RED_LIST_BCAST:%.+]] = bitcast{{.+}} [[RED_LIST]] to
// CHECK: [[K_RED_RET:%.+]] = call{{.+}} @__kmpc_reduce_nowait({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[RED_LIST_BCAST]], {{.+}} [[RED_FUN:@.+]], {{.+}} [[RED_VAR]])
// CHECK: switch{{.+}} [[K_RED_RET]], label{{.+}} [
// CHECK: {{.+}}, label %[[CASE1:.+]]
// CHECK: {{.+}}, label %[[CASE2:.+]]
// CHECK: ]
// CHECK: [[CASE1]]:
// CHECK-DAG: [[SIVAR_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_REF]],
// CHECK-DAG: [[SIVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_PRIV]],
// CHECK-DAG: [[SIVAR_INC:%.+]] = add{{.+}} [[SIVAR_VAL]], [[SIVAR_PRIV_VAL]]
// CHECK: store{{.+}} [[SIVAR_INC]], {{.+}} [[SIVAR_REF]],
// CHECK: call void @__kmpc_end_reduce_nowait({{.+}}, {{.+}}, {{.+}} [[RED_VAR]])
// CHECK: br
// CHECK: [[CASE2]]:
// CHECK-DAG: [[SIVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[SIVAR_PRIV]],
// CHECK-DAG: [[ATOMIC_RES:%.+]] = atomicrmw add{{.+}} [[SIVAR_REF]], {{.+}} [[SIVAR_PRIV_VAL]] monotonic, align {{.+}}
// CHECK: br

// CHECK: define{{.*}} i{{[0-9]+}} @[[TMAIN_INT]]()
// CHECK: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 1,
// CHECK: call void @[[TOFFL1:.+]]({{.+}})
// CHECK:  ret

// CHECK: define{{.*}} void @[[TOFFL1]](i{{64|32}}*{{.*}} [[TVAR_ARG:%.+]])
// CHECK: [[TVAR_ADDR:%.+]] = alloca i{{.+}},
// CHECK: store{{.+}} [[TVAR_ARG]], {{.+}} [[TVAR_ADDR]],
// CHECK: [[TVAR_VAL:%.+]] = load{{.+}}, {{.+}} [[TVAR_ADDR]]
// CHECK: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[TOUTL1:.+]] to {{.+}}, {{.+}} [[TVAR_VAL]])
// CHECK: ret void

// CHECK: define internal void @[[TOUTL1]]({{.+}}, {{.+}}, {{.+}} [[TVAR_ARG:%.+]])
// Skip global and bound tid vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: [[TVAR_ADDR:%.+]] = alloca i{{.+}},
// CHECK: [[TVAR_PRIV:%.+]] = alloca i{{.+}},
// CHECK: [[RED_LIST:%.+]] = alloca [1 x {{.+}}],
// CHECK: store{{.+}} [[TVAR_ARG]], {{.+}} [[TVAR_ADDR]],
// CHECK: [[TVAR_REF:%.+]] = load {{.+}} [[TVAR_ADDR]],
// CHECK: store{{.+}} 0, {{.+}} [[TVAR_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[TPAR_OUTL:.+]] to
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: [[RED_LIST_GEP:%.+]] = getelementptr{{.+}} [[RED_LIST]],
// CHECK: [[TVAR_PRIV_CAST:%.+]] = bitcast{{.+}} [[TVAR_PRIV]] to
// CHECK: store{{.+}} [[TVAR_PRIV_CAST]], {{.+}} [[RED_LIST_GEP]],
// CHECK: [[RED_LIST_BCAST:%.+]] = bitcast{{.+}} [[RED_LIST]] to
// CHECK: [[K_RED_RET:%.+]] = call{{.+}} @__kmpc_reduce_nowait({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[RED_LIST_BCAST]], {{.+}} [[RED_FUN:@.+]], {{.+}} [[RED_VAR]])
// CHECK: switch{{.+}} [[K_RED_RET]], label{{.+}} [
// CHECK: {{.+}}, label %[[CASE1:.+]]
// CHECK: {{.+}}, label %[[CASE2:.+]]
// CHECK: ]
// CHECK: [[CASE1]]:
// CHECK-DAG: [[TVAR_VAL:%.+]] = load{{.+}}, {{.+}} [[TVAR_REF]],
// CHECK-DAG: [[TVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[TVAR_PRIV]],
// CHECK-DAG: [[TVAR_INC:%.+]] = add{{.+}} [[TVAR_VAL]], [[TVAR_PRIV_VAL]]
// CHECK: store{{.+}} [[TVAR_INC]], {{.+}} [[TVAR_REF]],
// CHECK: call void @__kmpc_end_reduce_nowait({{.+}}, {{.+}}, {{.+}} [[RED_VAR]])
// CHECK: br
// CHECK: [[CASE2]]:
// CHECK-DAG: [[TVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[TVAR_PRIV]],
// CHECK-DAG: [[ATOMIC_RES:%.+]] = atomicrmw add{{.+}} [[TVAR_REF]], {{.+}} [[TVAR_PRIV_VAL]] monotonic, align {{.+}}
// CHECK: br

// CHECK: define internal void @[[TPAR_OUTL]]({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[TVAR_ARG:%.+]])
// Skip global and bound tid vars, and prev lb and ub vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[TVAR_ADDR:%.+]] = alloca i{{.+}},
// skip loop vars
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: alloca i32,
// CHECK: [[TVAR_PRIV:%.+]] = alloca i{{.+}},
// CHECK: [[RED_LIST:%.+]] = alloca [1 x {{.+}}],
// CHECK: store{{.+}} [[TVAR_ARG]], {{.+}} [[TVAR_ADDR]],
// CHECK: [[TVAR_REF:%.+]] = load {{.+}} [[TVAR_ADDR]],
// CHECK: store{{.+}} 0, {{.+}} [[TVAR_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: store{{.+}}, {{.+}} [[TVAR_PRIV]],
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: [[RED_LIST_GEP:%.+]] = getelementptr{{.+}} [[RED_LIST]],
// CHECK: [[TVAR_PRIV_CAST:%.+]] = bitcast{{.+}} [[TVAR_PRIV]] to
// CHECK: store{{.+}} [[TVAR_PRIV_CAST]], {{.+}} [[RED_LIST_GEP]],
// CHECK: [[RED_LIST_BCAST:%.+]] = bitcast{{.+}} [[RED_LIST]] to
// CHECK: [[K_RED_RET:%.+]] = call{{.+}} @__kmpc_reduce_nowait({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}} [[RED_LIST_BCAST]], {{.+}} [[RED_FUN:@.+]], {{.+}} [[RED_VAR]])
// CHECK: switch{{.+}} [[K_RED_RET]], label{{.+}} [
// CHECK: {{.+}}, label %[[CASE1:.+]]
// CHECK: {{.+}}, label %[[CASE2:.+]]
// CHECK: ]
// CHECK: [[CASE1]]:
// CHECK-DAG: [[TVAR_VAL:%.+]] = load{{.+}}, {{.+}} [[TVAR_REF]],
// CHECK-DAG: [[TVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[TVAR_PRIV]],
// CHECK-DAG: [[TVAR_INC:%.+]] = add{{.+}} [[TVAR_VAL]], [[TVAR_PRIV_VAL]]
// CHECK: store{{.+}} [[TVAR_INC]], {{.+}} [[TVAR_REF]],
// CHECK: call void @__kmpc_end_reduce_nowait({{.+}}, {{.+}}, {{.+}} [[RED_VAR]])
// CHECK: br
// CHECK: [[CASE2]]:
// CHECK-DAG: [[TVAR_PRIV_VAL:%.+]] = load{{.+}}, {{.+}} [[TVAR_PRIV]],
// CHECK-DAG: [[ATOMIC_RES:%.+]] = atomicrmw add{{.+}} [[TVAR_REF]], {{.+}} [[TVAR_PRIV_VAL]] monotonic, align {{.+}}
// CHECK: br
#endif
