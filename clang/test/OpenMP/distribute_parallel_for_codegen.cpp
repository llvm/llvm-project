// Test host code gen
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32
// RUN: %clang_cc1 -DLAMBDA -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1  -verify -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -verify -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1  -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1  -verify -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -verify -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER


template <typename T>
T tmain() {
  T *a, *b, *c;
  int n = 10000;
  int ch = 100;

  // no schedule clauses
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for
  for (int i = 0; i < n; ++i) {
    #pragma omp cancel for
    a[i] = b[i] + c[i];
  }

  // dist_schedule: static no chunk
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for dist_schedule(static)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
  }

  // dist_schedule: static chunk
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for dist_schedule(static, ch)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
  }

  // schedule: static no chunk
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
  }

  // schedule: static chunk
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for schedule(static, ch)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
  }

  // schedule: dynamic no chunk
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for schedule(dynamic)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
  }

  // schedule: dynamic chunk
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for schedule(dynamic, ch)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
  }

  return T();
}

int main() {
  double *a, *b, *c;
  int n = 10000;
  int ch = 100;

#ifdef LAMBDA
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{.*}} void [[OUTER_LAMBDA:@.+]](
  [&]() {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](

    // LAMBDA: call i{{[0-9]+}} @__tgt_target_teams(
    // LAMBDA: call void [[OFFLOADING_FUN_1:@.+]](

    // LAMBDA: call i{{[0-9]+}} @__tgt_target_teams(
    // LAMBDA: call void [[OFFLOADING_FUN_2:@.+]](

    // LAMBDA: call i{{[0-9]+}} @__tgt_target_teams(
    // LAMBDA: call void [[OFFLOADING_FUN_3:@.+]](

    // LAMBDA: call i{{[0-9]+}} @__tgt_target_teams(
    // LAMBDA: call void [[OFFLOADING_FUN_4:@.+]](

    // LAMBDA: call i{{[0-9]+}} @__tgt_target_teams(
    // LAMBDA: call void [[OFFLOADING_FUN_5:@.+]](

    // LAMBDA: call i{{[0-9]+}} @__tgt_target_teams(
    // LAMBDA: call void [[OFFLOADING_FUN_6:@.+]](

    // LAMBDA: call i{{[0-9]+}} @__tgt_target_teams(
    // LAMBDA: call void [[OFFLOADING_FUN_7:@.+]](

    // no schedule clauses
    #pragma omp target
    #pragma omp teams
    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN_1]](
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_1:@.+]] to {{.+}})

    #pragma omp distribute parallel for
    for (int i = 0; i < n; ++i) {
      a[i] = b[i] + c[i];
      // LAMBDA: define{{.+}} void [[OMP_OUTLINED_1]](
      // LAMBDA-DAG: [[OMP_IV:%.omp.iv]] = alloca
      // LAMBDA-DAG: [[OMP_LB:%.omp.comb.lb]] = alloca
      // LAMBDA-DAG: [[OMP_UB:%.omp.comb.ub]] = alloca
      // LAMBDA-DAG: [[OMP_ST:%.omp.stride]] = alloca

      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,

      // check EUB for distribute
      // LAMBDA-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
      // LAMBDA: [[NUM_IT_1:%.+]] = load{{.+}},
      // LAMBDA-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], [[NUM_IT_1]]
      // LAMBDA: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
      // LAMBDA-DAG: [[EUB_TRUE]]:
      // LAMBDA: [[NUM_IT_2:%.+]] = load{{.+}},
      // LAMBDA: br label %[[EUB_END:.+]]
      // LAMBDA-DAG: [[EUB_FALSE]]:
      // LAMBDA: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
      // LAMBDA: br label %[[EUB_END]]
      // LAMBDA-DAG: [[EUB_END]]:
      // LAMBDA-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ [[NUM_IT_2]], %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
      // LAMBDA: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

      // initialize omp.iv
      // LAMBDA: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
      // LAMBDA: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],
      // LAMBDA: br label %[[OMP_JUMP_BACK:.+]]

      // check exit condition
      // LAMBDA: [[OMP_JUMP_BACK]]:
      // LAMBDA-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_UB_VAL_3:%.+]] = load {{.+}} [[OMP_UB]],
      // LAMBDA: [[CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_IV_VAL_1]], [[OMP_UB_VAL_3]]
      // LAMBDA: br {{.+}} [[CMP_IV_UB]], label %[[DIST_BODY:.+]], label %[[DIST_END:.+]]

      // check that PrevLB and PrevUB are passed to the 'for'
      // LAMBDA: [[DIST_BODY]]:
      // LAMBDA-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
      // LAMBDA-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to
      // LAMBDA-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
      // LAMBDA-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to
      // check that distlb and distub are properly passed to fork_call
      // LAMBDA-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_1:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB]], i{{[0-9]+}} [[OMP_PREV_UB]], {{.+}})
      // LAMBDA-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_1:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_EXT]], i{{[0-9]+}} [[OMP_PREV_UB_EXT]], {{.+}})
      // LAMBDA: br label %[[DIST_INC:.+]]

      // increment by stride (distInc - 'parallel for' executes the whole chunk) and latch
      // LAMBDA: [[DIST_INC]]:
      // LAMBDA-DAG: [[OMP_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
      // LAMBDA: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_2]], [[OMP_ST_VAL_1]]
      // LAMBDA: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
      // LAMBDA: br label %[[OMP_JUMP_BACK]]

      // LAMBDA-DAG: call void @__kmpc_for_static_fini(
      // LAMBDA: ret

      // implementation of 'parallel for'
      // LAMBDA: define{{.+}} void [[OMP_PARFOR_OUTLINED_1]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})

      // LAMBDA-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},

      // initialize lb and ub to PrevLB and PrevUB
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
      // LAMBDA-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
      // LAMBDA-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 34, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})

      // PrevEUB is only used when 'for' has a chunked schedule, otherwise EUB is used
      // In this case we use EUB
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
      // LAMBDA: [[PF_NUM_IT_1:%.+]] = load{{.+}},
      // LAMBDA-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_NUM_IT_1]]
      // LAMBDA: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
      // LAMBDA: [[PF_EUB_TRUE]]:
      // LAMBDA: [[PF_NUM_IT_2:%.+]] = load{{.+}},
      // LAMBDA: br label %[[PF_EUB_END:.+]]
      // LAMBDA-DAG: [[PF_EUB_FALSE]]:
      // LAMBDA: [[OMP_PF_UB_VAL2:%.+]] = load{{.+}} [[OMP_PF_UB]],
      // LAMBDA: br label %[[PF_EUB_END]]
      // LAMBDA-DAG: [[PF_EUB_END]]:
      // LAMBDA-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_NUM_IT_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL2]], %[[PF_EUB_FALSE]] ]
      // LAMBDA: store{{.+}} [[PF_EUB_RES]],{{.+}}  [[OMP_PF_UB]],

      // initialize omp.iv
      // LAMBDA: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
      // LAMBDA: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label %[[OMP_PF_JUMP_BACK:.+]]

      // check exit condition
      // LAMBDA: [[OMP_PF_JUMP_BACK]]:
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load {{.+}} [[OMP_PF_IV]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load {{.+}} [[OMP_PF_UB]],
      // LAMBDA: [[PF_CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
      // LAMBDA: br {{.+}} [[PF_CMP_IV_UB]], label %[[PF_BODY:.+]], label %[[PF_END:.+]]

      // check that PrevLB and PrevUB are passed to the 'for'
      // LAMBDA: [[PF_BODY]]:
      // LAMBDA-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label {{.+}}

      // check stride 1 for 'for' in 'distribute parallel for'
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: [[OMP_PF_IV_INC:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_2]], 1
      // LAMBDA: store{{.+}} [[OMP_PF_IV_INC]], {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label %[[OMP_PF_JUMP_BACK]]

      // LAMBDA-DAG: call void @__kmpc_for_static_fini(
      // LAMBDA: ret

      [&]() {
	a[i] = b[i] + c[i];
      }();
    }

    // dist_schedule: static no chunk (same sa default - no dist_schedule)
    #pragma omp target
    #pragma omp teams
    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN_2]](
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_2:@.+]] to {{.+}})

    #pragma omp distribute parallel for dist_schedule(static)
    for (int i = 0; i < n; ++i) {
      a[i] = b[i] + c[i];
      // LAMBDA: define{{.+}} void [[OMP_OUTLINED_2]](
      // LAMBDA-DAG: [[OMP_IV:%.omp.iv]] = alloca
      // LAMBDA-DAG: [[OMP_LB:%.omp.comb.lb]] = alloca
      // LAMBDA-DAG: [[OMP_UB:%.omp.comb.ub]] = alloca
      // LAMBDA-DAG: [[OMP_ST:%.omp.stride]] = alloca

      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,

      // check EUB for distribute
      // LAMBDA-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
      // LAMBDA: [[NUM_IT_1:%.+]] = load{{.+}},
      // LAMBDA-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], [[NUM_IT_1]]
      // LAMBDA: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
      // LAMBDA-DAG: [[EUB_TRUE]]:
      // LAMBDA: [[NUM_IT_2:%.+]] = load{{.+}},
      // LAMBDA: br label %[[EUB_END:.+]]
      // LAMBDA-DAG: [[EUB_FALSE]]:
      // LAMBDA: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
      // LAMBDA: br label %[[EUB_END]]
      // LAMBDA-DAG: [[EUB_END]]:
      // LAMBDA-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ [[NUM_IT_2]], %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
      // LAMBDA: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

      // initialize omp.iv
      // LAMBDA: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
      // LAMBDA: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],
      // LAMBDA: br label %[[OMP_JUMP_BACK:.+]]

      // check exit condition
      // LAMBDA: [[OMP_JUMP_BACK]]:
      // LAMBDA-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_UB_VAL_3:%.+]] = load {{.+}} [[OMP_UB]],
      // LAMBDA: [[CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_IV_VAL_1]], [[OMP_UB_VAL_3]]
      // LAMBDA: br {{.+}} [[CMP_IV_UB]], label %[[DIST_BODY:.+]], label %[[DIST_END:.+]]

      // check that PrevLB and PrevUB are passed to the 'for'
      // LAMBDA: [[DIST_BODY]]:
      // LAMBDA-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
      // LAMBDA-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to
      // LAMBDA-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
      // LAMBDA-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to
      // check that distlb and distub are properly passed to fork_call
      // LAMBDA-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_2:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_EXT]], i{{[0-9]+}} [[OMP_PREV_UB_EXT]], {{.+}})
      // LAMBDA-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_2:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB]], i{{[0-9]+}} [[OMP_PREV_UB]], {{.+}})
      // LAMBDA: br label %[[DIST_INC:.+]]

      // increment by stride (distInc - 'parallel for' executes the whole chunk) and latch
      // LAMBDA: [[DIST_INC]]:
      // LAMBDA-DAG: [[OMP_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
      // LAMBDA: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_2]], [[OMP_ST_VAL_1]]
      // LAMBDA: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
      // LAMBDA: br label %[[OMP_JUMP_BACK]]

      // LAMBDA-DAG: call void @__kmpc_for_static_fini(
      // LAMBDA: ret

      // implementation of 'parallel for'
      // LAMBDA: define{{.+}} void [[OMP_PARFOR_OUTLINED_2]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})

      // LAMBDA-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},

      // initialize lb and ub to PrevLB and PrevUB
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
      // LAMBDA-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
      // LAMBDA-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 34, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})

      // PrevEUB is only used when 'for' has a chunked schedule, otherwise EUB is used
      // In this case we use EUB
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
      // LAMBDA: [[PF_NUM_IT_1:%.+]] = load{{.+}},
      // LAMBDA-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_NUM_IT_1]]
      // LAMBDA: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
      // LAMBDA: [[PF_EUB_TRUE]]:
      // LAMBDA: [[PF_NUM_IT_2:%.+]] = load{{.+}},
      // LAMBDA: br label %[[PF_EUB_END:.+]]
      // LAMBDA-DAG: [[PF_EUB_FALSE]]:
      // LAMBDA: [[OMP_PF_UB_VAL2:%.+]] = load{{.+}} [[OMP_PF_UB]],
      // LAMBDA: br label %[[PF_EUB_END]]
      // LAMBDA-DAG: [[PF_EUB_END]]:
      // LAMBDA-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_NUM_IT_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL2]], %[[PF_EUB_FALSE]] ]
      // LAMBDA: store{{.+}} [[PF_EUB_RES]],{{.+}}  [[OMP_PF_UB]],

      // initialize omp.iv
      // LAMBDA: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
      // LAMBDA: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label %[[OMP_PF_JUMP_BACK:.+]]

      // check exit condition
      // LAMBDA: [[OMP_PF_JUMP_BACK]]:
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load {{.+}} [[OMP_PF_IV]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load {{.+}} [[OMP_PF_UB]],
      // LAMBDA: [[PF_CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
      // LAMBDA: br {{.+}} [[PF_CMP_IV_UB]], label %[[PF_BODY:.+]], label %[[PF_END:.+]]

      // check that PrevLB and PrevUB are passed to the 'for'
      // LAMBDA: [[PF_BODY]]:
      // LAMBDA-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label {{.+}}

      // check stride 1 for 'for' in 'distribute parallel for'
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: [[OMP_PF_IV_INC:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_2]], 1
      // LAMBDA: store{{.+}} [[OMP_PF_IV_INC]], {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label %[[OMP_PF_JUMP_BACK]]

      // LAMBDA-DAG: call void @__kmpc_for_static_fini(
      // LAMBDA: ret
      [&]() {
	a[i] = b[i] + c[i];
      }();
    }

    // dist_schedule: static chunk
    #pragma omp target
    #pragma omp teams
    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN_3]](
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}}* [[OMP_OUTLINED_3:@.+]] to {{.+}})

    #pragma omp distribute parallel for dist_schedule(static, ch)
    for (int i = 0; i < n; ++i) {
      a[i] = b[i] + c[i];
      // LAMBDA: define{{.+}} void [[OMP_OUTLINED_3]](
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: [[OMP_IV:%.+]] = alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: [[OMP_LB:%.+]] = alloca
      // LAMBDA: [[OMP_UB:%.+]] = alloca
      // LAMBDA: [[OMP_ST:%.+]] = alloca

      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 91,

      // check EUB for distribute
      // LAMBDA-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
      // LAMBDA: [[NUM_IT_1:%.+]] = load{{.+}}
      // LAMBDA-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], [[NUM_IT_1]]
      // LAMBDA: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
      // LAMBDA-DAG: [[EUB_TRUE]]:
      // LAMBDA: [[NUM_IT_2:%.+]] = load{{.+}},
      // LAMBDA: br label %[[EUB_END:.+]]
      // LAMBDA-DAG: [[EUB_FALSE]]:
      // LAMBDA: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
      // LAMBDA: br label %[[EUB_END]]
      // LAMBDA-DAG: [[EUB_END]]:
      // LAMBDA-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ [[NUM_IT_2]], %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
      // LAMBDA: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

      // initialize omp.iv
      // LAMBDA: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
      // LAMBDA: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],

      // check exit condition
      // LAMBDA-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_UB_VAL_3:%.+]] = load {{.+}}
      // LAMBDA-DAG: [[OMP_UB_VAL_3_PLUS_ONE:%.+]] = add {{.+}} [[OMP_UB_VAL_3]], 1
      // LAMBDA: [[CMP_IV_UB:%.+]] = icmp slt {{.+}} [[OMP_IV_VAL_1]], [[OMP_UB_VAL_3_PLUS_ONE]]
      // LAMBDA: br {{.+}} [[CMP_IV_UB]], label %[[DIST_INNER_LOOP_BODY:.+]], label %[[DIST_INNER_LOOP_END:.+]]

      // check that PrevLB and PrevUB are passed to the 'for'
      // LAMBDA: [[DIST_INNER_LOOP_BODY]]:
      // LAMBDA-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
      // LAMBDA-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to {{.+}}
      // LAMBDA-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
      // LAMBDA-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to {{.+}}
      // check that distlb and distub are properly passed to fork_call
      // LAMBDA-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_3:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_EXT]], i{{[0-9]+}} [[OMP_PREV_UB_EXT]], {{.+}})
      // LAMBDA-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_3:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB]], i{{[0-9]+}} [[OMP_PREV_UB]], {{.+}})
      // LAMBDA: br label %[[DIST_INNER_LOOP_INC:.+]]

      // check DistInc
      // LAMBDA: [[DIST_INNER_LOOP_INC]]:
      // LAMBDA-DAG: [[OMP_IV_VAL_3:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
      // LAMBDA: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_3]], [[OMP_ST_VAL_1]]
      // LAMBDA: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_LB_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_LB]],
      // LAMBDA-DAG: [[OMP_ST_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_ST]],
      // LAMBDA-DAG: [[OMP_LB_NEXT:%.+]] = add{{.+}} [[OMP_LB_VAL_2]], [[OMP_ST_VAL_2]]
      // LAMBDA: store{{.+}} [[OMP_LB_NEXT]], {{.+}}* [[OMP_LB]],
      // LAMBDA-DAG: [[OMP_UB_VAL_5:%.+]] = load{{.+}}, {{.+}} [[OMP_UB]],
      // LAMBDA-DAG: [[OMP_ST_VAL_3:%.+]] = load{{.+}}, {{.+}} [[OMP_ST]],
      // LAMBDA-DAG: [[OMP_UB_NEXT:%.+]] = add{{.+}} [[OMP_UB_VAL_5]], [[OMP_ST_VAL_3]]
      // LAMBDA: store{{.+}} [[OMP_UB_NEXT]], {{.+}}* [[OMP_UB]],

      // Update UB
      // LAMBDA-DAG: [[OMP_UB_VAL_6:%.+]] = load{{.+}}, {{.+}} [[OMP_UB]],
      // LAMBDA: [[OMP_EXPR_VAL:%.+]] = load{{.+}}, {{.+}}
      // LAMBDA-DAG: [[CMP_UB_NUM_IT_1:%.+]] = icmp sgt {{.+}}[[OMP_UB_VAL_6]], [[OMP_EXPR_VAL]]
      // LAMBDA: br {{.+}} [[CMP_UB_NUM_IT_1]], label %[[EUB_TRUE_1:.+]], label %[[EUB_FALSE_1:.+]]
      // LAMBDA-DAG: [[EUB_TRUE_1]]:
      // LAMBDA: [[NUM_IT_3:%.+]] = load{{.+}}
      // LAMBDA: br label %[[EUB_END_1:.+]]
      // LAMBDA-DAG: [[EUB_FALSE_1]]:
      // LAMBDA: [[OMP_UB_VAL3:%.+]] = load{{.+}} [[OMP_UB]],
      // LAMBDA: br label %[[EUB_END_1]]
      // LAMBDA-DAG: [[EUB_END_1]]:
      // LAMBDA-DAG: [[EUB_RES_1:%.+]] = phi{{.+}} [ [[NUM_IT_3]], %[[EUB_TRUE_1]] ], [ [[OMP_UB_VAL3]], %[[EUB_FALSE_1]] ]
      // LAMBDA: store{{.+}} [[EUB_RES_1]], {{.+}}* [[OMP_UB]],

      // Store LB in IV
      // LAMBDA-DAG: [[OMP_LB_VAL_3:%.+]] = load{{.+}}, {{.+}} [[OMP_LB]],
      // LAMBDA: store{{.+}} [[OMP_LB_VAL_3]], {{.+}}* [[OMP_IV]],

      // LAMBDA: [[DIST_INNER_LOOP_END]]:
      // LAMBDA: br label %[[LOOP_EXIT:.+]]

      // loop exit
      // LAMBDA: [[LOOP_EXIT]]:
      // LAMBDA-DAG: call void @__kmpc_for_static_fini(
      // LAMBDA: ret

      // skip implementation of 'parallel for': using default scheduling and was tested above
      [&]() {
	a[i] = b[i] + c[i];
      }();
    }

    // schedule: static no chunk
    #pragma omp target
    #pragma omp teams
    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN_4]](
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_4:@.+]] to {{.+}})

    #pragma omp distribute parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      a[i] = b[i] + c[i];
      // LAMBDA: define{{.+}} void [[OMP_OUTLINED_4]](
      // LAMBDA-DAG: [[OMP_IV:%.omp.iv]] = alloca
      // LAMBDA-DAG: [[OMP_LB:%.omp.comb.lb]] = alloca
      // LAMBDA-DAG: [[OMP_UB:%.omp.comb.ub]] = alloca
      // LAMBDA-DAG: [[OMP_ST:%.omp.stride]] = alloca

      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
      // LAMBDA: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_4:@.+]] to {{.+}},
      // skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
      // LAMBDA: ret

      // 'parallel for' implementation is the same as the case without schedule clase (static no chunk is the default)
      // LAMBDA: define{{.+}} void [[OMP_PARFOR_OUTLINED_4]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})

      // LAMBDA-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},

      // initialize lb and ub to PrevLB and PrevUB
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
      // LAMBDA-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
      // LAMBDA-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 34, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})

      // PrevEUB is only used when 'for' has a chunked schedule, otherwise EUB is used
      // In this case we use EUB
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
      // LAMBDA: [[PF_NUM_IT_1:%.+]] = load{{.+}},
      // LAMBDA-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_NUM_IT_1]]
      // LAMBDA: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
      // LAMBDA: [[PF_EUB_TRUE]]:
      // LAMBDA: [[PF_NUM_IT_2:%.+]] = load{{.+}},
      // LAMBDA: br label %[[PF_EUB_END:.+]]
      // LAMBDA-DAG: [[PF_EUB_FALSE]]:
      // LAMBDA: [[OMP_PF_UB_VAL2:%.+]] = load{{.+}} [[OMP_PF_UB]],
      // LAMBDA: br label %[[PF_EUB_END]]
      // LAMBDA-DAG: [[PF_EUB_END]]:
      // LAMBDA-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_NUM_IT_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL2]], %[[PF_EUB_FALSE]] ]
      // LAMBDA: store{{.+}} [[PF_EUB_RES]],{{.+}}  [[OMP_PF_UB]],

      // initialize omp.iv
      // LAMBDA: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
      // LAMBDA: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label %[[OMP_PF_JUMP_BACK:.+]]

      // check exit condition
      // LAMBDA: [[OMP_PF_JUMP_BACK]]:
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load {{.+}} [[OMP_PF_IV]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load {{.+}} [[OMP_PF_UB]],
      // LAMBDA: [[PF_CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
      // LAMBDA: br {{.+}} [[PF_CMP_IV_UB]], label %[[PF_BODY:.+]], label %[[PF_END:.+]]

      // check that PrevLB and PrevUB are passed to the 'for'
      // LAMBDA: [[PF_BODY]]:
      // LAMBDA-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label {{.+}}

      // check stride 1 for 'for' in 'distribute parallel for'
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: [[OMP_PF_IV_INC:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_2]], 1
      // LAMBDA: store{{.+}} [[OMP_PF_IV_INC]], {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label %[[OMP_PF_JUMP_BACK]]

      // LAMBDA-DAG: call void @__kmpc_for_static_fini(
      // LAMBDA: ret

      [&]() {
	a[i] = b[i] + c[i];
      }();
    }

    // schedule: static chunk
    #pragma omp target
    #pragma omp teams
    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN_5]](
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}}* [[OMP_OUTLINED_5:@.+]] to {{.+}})

    #pragma omp distribute parallel for schedule(static, ch)
    for (int i = 0; i < n; ++i) {
      a[i] = b[i] + c[i];
      // LAMBDA: define{{.+}} void [[OMP_OUTLINED_5]](
      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
      // LAMBDA: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_5:@.+]] to {{.+}},
      // skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
      // LAMBDA: ret

      // 'parallel for' implementation using outer and inner loops and PrevEUB
      // LAMBDA: define{{.+}} void [[OMP_PARFOR_OUTLINED_5]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}})
      // LAMBDA-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_ST:%.omp.stride]] = alloca{{.+}},

      // initialize lb and ub to PrevLB and PrevUB
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
      // LAMBDA-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
      // LAMBDA-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})
      // LAMBDA: br label %[[OMP_PF_OUTER_LOOP_HEADER:.+]]

      // check PrevEUB (using PrevUB instead of NumIt as upper bound)
      // LAMBDA: [[OMP_PF_OUTER_LOOP_HEADER]]:
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
      // LAMBDA-64-DAG: [[OMP_PF_UB_VAL_CONV:%.+]] = sext{{.+}} [[OMP_PF_UB_VAL_1]] to
      // LAMBDA: [[PF_PREV_UB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
      // LAMBDA-64-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_CONV]], [[PF_PREV_UB_VAL_1]]
      // LAMBDA-32-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_PREV_UB_VAL_1]]
      // LAMBDA: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
      // LAMBDA: [[PF_EUB_TRUE]]:
      // LAMBDA: [[PF_PREV_UB_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
      // LAMBDA: br label %[[PF_EUB_END:.+]]
      // LAMBDA-DAG: [[PF_EUB_FALSE]]:
      // LAMBDA: [[OMP_PF_UB_VAL_2:%.+]] = load{{.+}} [[OMP_PF_UB]],
      // LAMBDA-64: [[OMP_PF_UB_VAL_2_CONV:%.+]] = sext{{.+}} [[OMP_PF_UB_VAL_2]] to
      // LAMBDA: br label %[[PF_EUB_END]]
      // LAMBDA-DAG: [[PF_EUB_END]]:
      // LAMBDA-64-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_PREV_UB_VAL_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL_2_CONV]], %[[PF_EUB_FALSE]] ]
      // LAMBDA-32-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_PREV_UB_VAL_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL_2]], %[[PF_EUB_FALSE]] ]
      // LAMBDA-64-DAG: [[PF_EUB_RES_CONV:%.+]] = trunc{{.+}} [[PF_EUB_RES]] to
      // LAMBDA-64: store{{.+}} [[PF_EUB_RES_CONV]],{{.+}}  [[OMP_PF_UB]],
      // LAMBDA-32: store{{.+}} [[PF_EUB_RES]], {{.+}} [[OMP_PF_UB]],

      // initialize omp.iv (IV = LB)
      // LAMBDA: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
      // LAMBDA: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],

      // outer loop: while (IV < UB) {
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
      // LAMBDA: [[PF_CMP_IV_UB_1:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
      // LAMBDA: br{{.+}} [[PF_CMP_IV_UB_1]], label %[[OMP_PF_OUTER_LOOP_BODY:.+]], label %[[OMP_PF_OUTER_LOOP_END:.+]]

      // LAMBDA: [[OMP_PF_OUTER_LOOP_BODY]]:
      // LAMBDA: br label %[[OMP_PF_INNER_FOR_HEADER:.+]]

      // LAMBDA: [[OMP_PF_INNER_FOR_HEADER]]:
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_4:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
      // LAMBDA: [[PF_CMP_IV_UB_2:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_2]], [[OMP_PF_UB_VAL_4]]
      // LAMBDA: br{{.+}} [[PF_CMP_IV_UB_2]], label %[[OMP_PF_INNER_LOOP_BODY:.+]], label %[[OMP_PF_INNER_LOOP_END:.+]]

      // LAMBDA: [[OMP_PF_INNER_LOOP_BODY]]:
      // LAMBDA-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // skip body branch
      // LAMBDA: br{{.+}}
      // LAMBDA: br label %[[OMP_PF_INNER_LOOP_INC:.+]]

      // IV = IV + 1 and inner loop latch
      // LAMBDA: [[OMP_PF_INNER_LOOP_INC]]:
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_PF_NEXT_IV:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_3]], 1
      // LAMBDA-DAG: store{{.+}} [[OMP_PF_NEXT_IV]], {{.+}}* [[OMP_IV]],
      // LAMBDA: br label %[[OMP_PF_INNER_FOR_HEADER]]

      // check NextLB and NextUB
      // LAMBDA: [[OMP_PF_INNER_LOOP_END]]:
      // LAMBDA: br label %[[OMP_PF_OUTER_LOOP_INC:.+]]

      // LAMBDA: [[OMP_PF_OUTER_LOOP_INC]]:
      // LAMBDA-DAG: [[OMP_PF_LB_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
      // LAMBDA-DAG: [[OMP_PF_ST_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_ST]],
      // LAMBDA-DAG: [[OMP_PF_LB_NEXT:%.+]] = add{{.+}} [[OMP_PF_LB_VAL_2]], [[OMP_PF_ST_VAL_1]]
      // LAMBDA: store{{.+}} [[OMP_PF_LB_NEXT]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_5:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_UB]],
      // LAMBDA-DAG: [[OMP_PF_ST_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_ST]],
      // LAMBDA-DAG: [[OMP_PF_UB_NEXT:%.+]] = add{{.+}} [[OMP_PF_UB_VAL_5]], [[OMP_PF_ST_VAL_2]]
      // LAMBDA: store{{.+}} [[OMP_PF_UB_NEXT]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA: br label %[[OMP_PF_OUTER_LOOP_HEADER]]

      // LAMBDA: [[OMP_PF_OUTER_LOOP_END]]:
      // LAMBDA-DAG: call void @__kmpc_for_static_fini(
      // LAMBDA: ret
      [&]() {
	a[i] = b[i] + c[i];
      }();
    }

    // schedule: dynamic no chunk
    #pragma omp target
    #pragma omp teams
    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN_6]](
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_6:@.+]] to {{.+}})

    #pragma omp distribute parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
      a[i] = b[i] + c[i];
      // LAMBDA: define{{.+}} void [[OMP_OUTLINED_6]](
      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
      // LAMBDA: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_6:@.+]] to {{.+}},
      // skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
      // LAMBDA: ret

      // 'parallel for' implementation using outer and inner loops and PrevEUB
      // LAMBDA: define{{.+}} void [[OMP_PARFOR_OUTLINED_6]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})
      // LAMBDA-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_ST:%.omp.stride]] = alloca{{.+}},

      // initialize lb and ub to PrevLB and PrevUB
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
      // LAMBDA-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
      // LAMBDA-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA-DAG: [[OMP_PF_LB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_UB]],
      // LAMBDA: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, {{.+}} 35, {{.+}} [[OMP_PF_LB_VAL]], {{.+}} [[OMP_PF_UB_VAL]], {{.+}}, {{.+}})
      // LAMBDA: br label %[[OMP_PF_OUTER_LOOP_HEADER:.+]]

      // LAMBDA: [[OMP_PF_OUTER_LOOP_HEADER]]:
      // LAMBDA: [[IS_FIN:%.+]] = call{{.+}} @__kmpc_dispatch_next_4({{.+}}, {{.+}}, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]], {{.+}}* [[OMP_PF_ST]])
      // LAMBDA: [[IS_FIN_CMP:%.+]] = icmp{{.+}} [[IS_FIN]], 0
      // LAMBDA: br{{.+}} [[IS_FIN_CMP]], label %[[OMP_PF_OUTER_LOOP_BODY:.+]], label %[[OMP_PF_OUTER_LOOP_END:.+]]

      // initialize omp.iv (IV = LB)
      // LAMBDA: [[OMP_PF_OUTER_LOOP_BODY]]:
      // LAMBDA-DAG: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
      // LAMBDA-DAG: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label %[[OMP_PF_INNER_LOOP_HEADER:.+]]

      // LAMBDA: [[OMP_PF_INNER_LOOP_HEADER]]:
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_4:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
      // LAMBDA: [[PF_CMP_IV_UB_2:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_2]], [[OMP_PF_UB_VAL_4]]
      // LAMBDA: br{{.+}} [[PF_CMP_IV_UB_2]], label %[[OMP_PF_INNER_LOOP_BODY:.+]], label %[[OMP_PF_INNER_LOOP_END:.+]]

      // LAMBDA: [[OMP_PF_INNER_LOOP_BODY]]:
      // LAMBDA-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // skip body branch
      // LAMBDA: br{{.+}}
      // LAMBDA: br label %[[OMP_PF_INNER_LOOP_INC:.+]]

      // IV = IV + 1 and inner loop latch
      // LAMBDA: [[OMP_PF_INNER_LOOP_INC]]:
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_PF_NEXT_IV:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_3]], 1
      // LAMBDA-DAG: store{{.+}} [[OMP_PF_NEXT_IV]], {{.+}}* [[OMP_IV]],
      // LAMBDA: br label %[[OMP_PF_INNER_LOOP_HEADER]]

      // check NextLB and NextUB
      // LAMBDA: [[OMP_PF_INNER_LOOP_END]]:
      // LAMBDA: br label %[[OMP_PF_OUTER_LOOP_INC:.+]]

      // LAMBDA: [[OMP_PF_OUTER_LOOP_INC]]:
      // LAMBDA: br label %[[OMP_PF_OUTER_LOOP_HEADER]]

      // LAMBDA: [[OMP_PF_OUTER_LOOP_END]]:
      // LAMBDA: ret
      [&]() {
	a[i] = b[i] + c[i];
      }();
    }

    // schedule: dynamic chunk
    #pragma omp target
    #pragma omp teams
    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN_7]](
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}}* [[OMP_OUTLINED_7:@.+]] to {{.+}})

    #pragma omp distribute parallel for schedule(dynamic, ch)
    for (int i = 0; i < n; ++i) {
      a[i] = b[i] + c[i];
      // LAMBDA: define{{.+}} void [[OMP_OUTLINED_7]](
      // LAMBDA: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
      // LAMBDA: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_7:@.+]] to {{.+}},
      // skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
      // LAMBDA: ret

      // 'parallel for' implementation using outer and inner loops and PrevEUB
      // LAMBDA: define{{.+}} void [[OMP_PARFOR_OUTLINED_7]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}})
      // LAMBDA-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},
      // LAMBDA-DAG: [[OMP_PF_ST:%.omp.stride]] = alloca{{.+}},

      // initialize lb and ub to PrevLB and PrevUB
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
      // LAMBDA-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
      // LAMBDA-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
      // LAMBDA-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
      // LAMBDA-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
      // LAMBDA-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
      // LAMBDA-DAG: [[OMP_PF_LB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_UB]],
      // LAMBDA: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, {{.+}} 35, {{.+}} [[OMP_PF_LB_VAL]], {{.+}} [[OMP_PF_UB_VAL]], {{.+}}, {{.+}})
      // LAMBDA: br label %[[OMP_PF_OUTER_LOOP_HEADER:.+]]

      // LAMBDA: [[OMP_PF_OUTER_LOOP_HEADER]]:
      // LAMBDA: [[IS_FIN:%.+]] = call{{.+}} @__kmpc_dispatch_next_4({{.+}}, {{.+}}, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]], {{.+}}* [[OMP_PF_ST]])
      // LAMBDA: [[IS_FIN_CMP:%.+]] = icmp{{.+}} [[IS_FIN]], 0
      // LAMBDA: br{{.+}} [[IS_FIN_CMP]], label %[[OMP_PF_OUTER_LOOP_BODY:.+]], label %[[OMP_PF_OUTER_LOOP_END:.+]]

      // initialize omp.iv (IV = LB)
      // LAMBDA: [[OMP_PF_OUTER_LOOP_BODY]]:
      // LAMBDA-DAG: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
      // LAMBDA-DAG: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
      // LAMBDA: br label %[[OMP_PF_INNER_LOOP_HEADER:.+]]

      // LAMBDA: [[OMP_PF_INNER_LOOP_HEADER]]:
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // LAMBDA-DAG: [[OMP_PF_UB_VAL_4:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
      // LAMBDA: [[PF_CMP_IV_UB_2:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_2]], [[OMP_PF_UB_VAL_4]]
      // LAMBDA: br{{.+}} [[PF_CMP_IV_UB_2]], label %[[OMP_PF_INNER_LOOP_BODY:.+]], label %[[OMP_PF_INNER_LOOP_END:.+]]

      // LAMBDA: [[OMP_PF_INNER_LOOP_BODY]]:
      // LAMBDA-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
      // skip body branch
      // LAMBDA: br{{.+}}
      // LAMBDA: br label %[[OMP_PF_INNER_LOOP_INC:.+]]

      // IV = IV + 1 and inner loop latch
      // LAMBDA: [[OMP_PF_INNER_LOOP_INC]]:
      // LAMBDA-DAG: [[OMP_PF_IV_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_IV]],
      // LAMBDA-DAG: [[OMP_PF_NEXT_IV:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_3]], 1
      // LAMBDA-DAG: store{{.+}} [[OMP_PF_NEXT_IV]], {{.+}}* [[OMP_IV]],
      // LAMBDA: br label %[[OMP_PF_INNER_LOOP_HEADER]]

      // check NextLB and NextUB
      // LAMBDA: [[OMP_PF_INNER_LOOP_END]]:
      // LAMBDA: br label %[[OMP_PF_OUTER_LOOP_INC:.+]]

      // LAMBDA: [[OMP_PF_OUTER_LOOP_INC]]:
      // LAMBDA: br label %[[OMP_PF_OUTER_LOOP_HEADER]]

      // LAMBDA: [[OMP_PF_OUTER_LOOP_END]]:
      // LAMBDA: ret
      [&]() {
	a[i] = b[i] + c[i];
      }();
    }
  }();
  return 0;
#else
  // CHECK-LABEL: @main

  // CHECK: call i{{[0-9]+}} @__tgt_target_teams(
  // CHECK: call void [[OFFLOADING_FUN_1:@.+]](

  // CHECK: call i{{[0-9]+}} @__tgt_target_teams(
  // CHECK: call void [[OFFLOADING_FUN_2:@.+]](

  // CHECK: call i{{[0-9]+}} @__tgt_target_teams(
  // CHECK: call void [[OFFLOADING_FUN_3:@.+]](

  // CHECK: call i{{[0-9]+}} @__tgt_target_teams(
  // CHECK: call void [[OFFLOADING_FUN_4:@.+]](

  // CHECK: call i{{[0-9]+}} @__tgt_target_teams(
  // CHECK: call void [[OFFLOADING_FUN_5:@.+]](

  // CHECK: call i{{[0-9]+}} @__tgt_target_teams(
  // CHECK: call void [[OFFLOADING_FUN_6:@.+]](

  // CHECK: call i{{[0-9]+}} @__tgt_target_teams(
  // CHECK: call void [[OFFLOADING_FUN_7:@.+]](

  // CHECK: call{{.+}} [[TMAIN:@.+]]()

  // no schedule clauses
  #pragma omp target
  #pragma omp teams
  // CHECK: define internal void [[OFFLOADING_FUN_1]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_1:@.+]] to {{.+}})

  #pragma omp distribute parallel for
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
    // CHECK: define{{.+}} void [[OMP_OUTLINED_1]](
    // CHECK-DAG: [[OMP_IV:%.omp.iv]] = alloca
    // CHECK-DAG: [[OMP_LB:%.omp.comb.lb]] = alloca
    // CHECK-DAG: [[OMP_UB:%.omp.comb.ub]] = alloca
    // CHECK-DAG: [[OMP_ST:%.omp.stride]] = alloca

    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,

    // check EUB for distribute
    // CHECK-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
    // CHECK: [[NUM_IT_1:%.+]] = load{{.+}},
    // CHECK-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], [[NUM_IT_1]]
    // CHECK: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
    // CHECK-DAG: [[EUB_TRUE]]:
    // CHECK: [[NUM_IT_2:%.+]] = load{{.+}},
    // CHECK: br label %[[EUB_END:.+]]
    // CHECK-DAG: [[EUB_FALSE]]:
    // CHECK: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
    // CHECK: br label %[[EUB_END]]
    // CHECK-DAG: [[EUB_END]]:
    // CHECK-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ [[NUM_IT_2]], %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
    // CHECK: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

    // initialize omp.iv
    // CHECK: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
    // CHECK: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],
    // CHECK: br label %[[OMP_JUMP_BACK:.+]]

    // check exit condition
    // CHECK: [[OMP_JUMP_BACK]]:
    // CHECK-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
    // CHECK-DAG: [[OMP_UB_VAL_3:%.+]] = load {{.+}} [[OMP_UB]],
    // CHECK: [[CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_IV_VAL_1]], [[OMP_UB_VAL_3]]
    // CHECK: br {{.+}} [[CMP_IV_UB]], label %[[DIST_BODY:.+]], label %[[DIST_END:.+]]

    // check that PrevLB and PrevUB are passed to the 'for'
    // CHECK: [[DIST_BODY]]:
    // CHECK-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
    // CHECK-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to {{.+}}
    // CHECK-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
    // CHECK-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to {{.+}}
    // check that distlb and distub are properly passed to fork_call
    // CHECK-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_1:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_EXT]], i{{[0-9]+}} [[OMP_PREV_UB_EXT]], {{.+}})
    // CHECK-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_1:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB]], i{{[0-9]+}} [[OMP_PREV_UB]], {{.+}})
    // CHECK: br label %[[DIST_INC:.+]]

    // increment by stride (distInc - 'parallel for' executes the whole chunk) and latch
    // CHECK: [[DIST_INC]]:
    // CHECK-DAG: [[OMP_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
    // CHECK-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
    // CHECK: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_2]], [[OMP_ST_VAL_1]]
    // CHECK: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
    // CHECK: br label %[[OMP_JUMP_BACK]]

    // CHECK-DAG: call void @__kmpc_for_static_fini(
    // CHECK: ret

    // implementation of 'parallel for'
    // CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_1]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})

    // CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},

    // initialize lb and ub to PrevLB and PrevUB
    // CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
    // CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
    // CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
    // CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 34, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})

    // PrevEUB is only used when 'for' has a chunked schedule, otherwise EUB is used
    // In this case we use EUB
    // CHECK-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
    // CHECK: [[PF_NUM_IT_1:%.+]] = load{{.+}},
    // CHECK-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_NUM_IT_1]]
    // CHECK: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
    // CHECK: [[PF_EUB_TRUE]]:
    // CHECK: [[PF_NUM_IT_2:%.+]] = load{{.+}},
    // CHECK: br label %[[PF_EUB_END:.+]]
    // CHECK-DAG: [[PF_EUB_FALSE]]:
    // CHECK: [[OMP_PF_UB_VAL2:%.+]] = load{{.+}} [[OMP_PF_UB]],
    // CHECK: br label %[[PF_EUB_END]]
    // CHECK-DAG: [[PF_EUB_END]]:
    // CHECK-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_NUM_IT_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL2]], %[[PF_EUB_FALSE]] ]
    // CHECK: store{{.+}} [[PF_EUB_RES]],{{.+}}  [[OMP_PF_UB]],

    // initialize omp.iv
    // CHECK: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
    // CHECK: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label %[[OMP_PF_JUMP_BACK:.+]]

    // check exit condition
    // CHECK: [[OMP_PF_JUMP_BACK]]:
    // CHECK-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load {{.+}} [[OMP_PF_IV]],
    // CHECK-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load {{.+}} [[OMP_PF_UB]],
    // CHECK: [[PF_CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
    // CHECK: br {{.+}} [[PF_CMP_IV_UB]], label %[[PF_BODY:.+]], label %[[PF_END:.+]]

    // check that PrevLB and PrevUB are passed to the 'for'
    // CHECK: [[PF_BODY]]:
    // CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label {{.+}}

    // check stride 1 for 'for' in 'distribute parallel for'
    // CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK: [[OMP_PF_IV_INC:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_2]], 1
    // CHECK: store{{.+}} [[OMP_PF_IV_INC]], {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label %[[OMP_PF_JUMP_BACK]]

    // CHECK-DAG: call void @__kmpc_for_static_fini(
    // CHECK: ret
  }

  // dist_schedule: static no chunk
  #pragma omp target
  #pragma omp teams
  // CHECK: define{{.+}} void [[OFFLOADING_FUN_2]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_2:@.+]] to {{.+}})

  #pragma omp distribute parallel for dist_schedule(static)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
    // CHECK: define{{.+}} void [[OMP_OUTLINED_2]](
    // CHECK-DAG: [[OMP_IV:%.omp.iv]] = alloca
    // CHECK-DAG: [[OMP_LB:%.omp.comb.lb]] = alloca
    // CHECK-DAG: [[OMP_UB:%.omp.comb.ub]] = alloca
    // CHECK-DAG: [[OMP_ST:%.omp.stride]] = alloca

    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,

    // check EUB for distribute
    // CHECK-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
    // CHECK: [[NUM_IT_1:%.+]] = load{{.+}},
    // CHECK-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], [[NUM_IT_1]]
    // CHECK: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
    // CHECK-DAG: [[EUB_TRUE]]:
    // CHECK: [[NUM_IT_2:%.+]] = load{{.+}},
    // CHECK: br label %[[EUB_END:.+]]
    // CHECK-DAG: [[EUB_FALSE]]:
    // CHECK: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
    // CHECK: br label %[[EUB_END]]
    // CHECK-DAG: [[EUB_END]]:
    // CHECK-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ [[NUM_IT_2]], %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
    // CHECK: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

    // initialize omp.iv
    // CHECK: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
    // CHECK: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],
    // CHECK: br label %[[OMP_JUMP_BACK:.+]]

    // check exit condition
    // CHECK: [[OMP_JUMP_BACK]]:
    // CHECK-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
    // CHECK-DAG: [[OMP_UB_VAL_3:%.+]] = load {{.+}} [[OMP_UB]],
    // CHECK: [[CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_IV_VAL_1]], [[OMP_UB_VAL_3]]
    // CHECK: br {{.+}} [[CMP_IV_UB]], label %[[DIST_BODY:.+]], label %[[DIST_END:.+]]

    // check that PrevLB and PrevUB are passed to the 'for'
    // CHECK: [[DIST_BODY]]:
    // CHECK-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
    // CHECK-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to {{.+}}
    // CHECK-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
    // CHECK-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to {{.+}}
    // check that distlb and distub are properly passed to fork_call
    // CHECK-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_2:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_EXT]], i{{[0-9]+}} [[OMP_PREV_UB_EXT]], {{.+}})
    // CHECK-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_2:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB]], i{{[0-9]+}} [[OMP_PREV_UB]], {{.+}})
    // CHECK: br label %[[DIST_INC:.+]]

    // increment by stride (distInc - 'parallel for' executes the whole chunk) and latch
    // CHECK: [[DIST_INC]]:
    // CHECK-DAG: [[OMP_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
    // CHECK-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
    // CHECK: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_2]], [[OMP_ST_VAL_1]]
    // CHECK: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
    // CHECK: br label %[[OMP_JUMP_BACK]]

    // CHECK-DAG: call void @__kmpc_for_static_fini(
    // CHECK: ret

    // implementation of 'parallel for'
    // CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_2]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})

    // CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},

    // initialize lb and ub to PrevLB and PrevUB
    // CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
    // CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
    // CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
    // CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 34, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})

    // PrevEUB is only used when 'for' has a chunked schedule, otherwise EUB is used
    // In this case we use EUB
    // CHECK-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
    // CHECK: [[PF_NUM_IT_1:%.+]] = load{{.+}},
    // CHECK-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_NUM_IT_1]]
    // CHECK: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
    // CHECK: [[PF_EUB_TRUE]]:
    // CHECK: [[PF_NUM_IT_2:%.+]] = load{{.+}},
    // CHECK: br label %[[PF_EUB_END:.+]]
    // CHECK-DAG: [[PF_EUB_FALSE]]:
    // CHECK: [[OMP_PF_UB_VAL2:%.+]] = load{{.+}} [[OMP_PF_UB]],
    // CHECK: br label %[[PF_EUB_END]]
    // CHECK-DAG: [[PF_EUB_END]]:
    // CHECK-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_NUM_IT_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL2]], %[[PF_EUB_FALSE]] ]
    // CHECK: store{{.+}} [[PF_EUB_RES]],{{.+}}  [[OMP_PF_UB]],

    // initialize omp.iv
    // CHECK: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
    // CHECK: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label %[[OMP_PF_JUMP_BACK:.+]]

    // check exit condition
    // CHECK: [[OMP_PF_JUMP_BACK]]:
    // CHECK-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load {{.+}} [[OMP_PF_IV]],
    // CHECK-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load {{.+}} [[OMP_PF_UB]],
    // CHECK: [[PF_CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
    // CHECK: br {{.+}} [[PF_CMP_IV_UB]], label %[[PF_BODY:.+]], label %[[PF_END:.+]]

    // check that PrevLB and PrevUB are passed to the 'for'
    // CHECK: [[PF_BODY]]:
    // CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label {{.+}}

    // check stride 1 for 'for' in 'distribute parallel for'
    // CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK: [[OMP_PF_IV_INC:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_2]], 1
    // CHECK: store{{.+}} [[OMP_PF_IV_INC]], {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label %[[OMP_PF_JUMP_BACK]]

    // CHECK-DAG: call void @__kmpc_for_static_fini(
    // CHECK: ret
  }

  // dist_schedule: static chunk
  #pragma omp target
  #pragma omp teams
  // CHECK: define{{.+}} void [[OFFLOADING_FUN_3]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}}* [[OMP_OUTLINED_3:@.+]] to {{.+}})

  #pragma omp distribute parallel for dist_schedule(static, ch)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
    // CHECK: define{{.+}} void [[OMP_OUTLINED_3]](
    // CHECK: alloca
    // CHECK: alloca
    // CHECK: alloca
    // CHECK: alloca
    // CHECK: alloca
    // CHECK: alloca
    // CHECK: alloca
    // CHECK: [[OMP_IV:%.+]] = alloca
    // CHECK: alloca
    // CHECK: alloca
    // CHECK: alloca
    // CHECK: alloca
    // CHECK: [[OMP_LB:%.+]] = alloca
    // CHECK: [[OMP_UB:%.+]] = alloca
    // CHECK: [[OMP_ST:%.+]] = alloca

    // unlike the previous tests, in this one we have a outer and inner loop for 'distribute'
    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 91,

    // check EUB for distribute
    // CHECK-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
    // CHECK: [[NUM_IT_1:%.+]] = load{{.+}}
    // CHECK-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], [[NUM_IT_1]]
    // CHECK: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
    // CHECK-DAG: [[EUB_TRUE]]:
    // CHECK: [[NUM_IT_2:%.+]] = load{{.+}},
    // CHECK: br label %[[EUB_END:.+]]
    // CHECK-DAG: [[EUB_FALSE]]:
    // CHECK: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
    // CHECK: br label %[[EUB_END]]
    // CHECK-DAG: [[EUB_END]]:
    // CHECK-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ [[NUM_IT_2]], %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
    // CHECK: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

    // initialize omp.iv
    // CHECK: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
    // CHECK: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],

    // check exit condition
    // CHECK-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
    // CHECK-DAG: [[OMP_UB_VAL_3:%.+]] = load {{.+}}
    // CHECK-DAG: [[OMP_UB_VAL_3_PLUS_ONE:%.+]] = add {{.+}} [[OMP_UB_VAL_3]], 1
    // CHECK: [[CMP_IV_UB:%.+]] = icmp slt {{.+}} [[OMP_IV_VAL_1]], [[OMP_UB_VAL_3_PLUS_ONE]]
    // CHECK: br {{.+}} [[CMP_IV_UB]], label %[[DIST_INNER_LOOP_BODY:.+]], label %[[DIST_INNER_LOOP_END:.+]]

    // check that PrevLB and PrevUB are passed to the 'for'
    // CHECK: [[DIST_INNER_LOOP_BODY]]:
    // CHECK-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
    // CHECK-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to {{.+}}
    // CHECK-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
    // CHECK-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to {{.+}}
    // check that distlb and distub are properly passed to fork_call
    // CHECK-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_3:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_EXT]], i{{[0-9]+}} [[OMP_PREV_UB_EXT]], {{.+}})
    // CHECK-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_3:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB]], i{{[0-9]+}} [[OMP_PREV_UB]], {{.+}})
    // CHECK: br label %[[DIST_INNER_LOOP_INC:.+]]

    // check DistInc
    // CHECK: [[DIST_INNER_LOOP_INC]]:
    // CHECK-DAG: [[OMP_IV_VAL_3:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
    // CHECK-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
    // CHECK: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_3]], [[OMP_ST_VAL_1]]
    // CHECK: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
    // CHECK-DAG: [[OMP_LB_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_LB]],
    // CHECK-DAG: [[OMP_ST_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_ST]],
    // CHECK-DAG: [[OMP_LB_NEXT:%.+]] = add{{.+}} [[OMP_LB_VAL_2]], [[OMP_ST_VAL_2]]
    // CHECK: store{{.+}} [[OMP_LB_NEXT]], {{.+}}* [[OMP_LB]],
    // CHECK-DAG: [[OMP_UB_VAL_5:%.+]] = load{{.+}}, {{.+}} [[OMP_UB]],
    // CHECK-DAG: [[OMP_ST_VAL_3:%.+]] = load{{.+}}, {{.+}} [[OMP_ST]],
    // CHECK-DAG: [[OMP_UB_NEXT:%.+]] = add{{.+}} [[OMP_UB_VAL_5]], [[OMP_ST_VAL_3]]
    // CHECK: store{{.+}} [[OMP_UB_NEXT]], {{.+}}* [[OMP_UB]],

    // Update UB
    // CHECK-DAG: [[OMP_UB_VAL_6:%.+]] = load{{.+}}, {{.+}} [[OMP_UB]],
    // CHECK: [[OMP_EXPR_VAL:%.+]] = load{{.+}}, {{.+}}
    // CHECK-DAG: [[CMP_UB_NUM_IT_1:%.+]] = icmp sgt {{.+}}[[OMP_UB_VAL_6]], [[OMP_EXPR_VAL]]
    // CHECK: br {{.+}} [[CMP_UB_NUM_IT_1]], label %[[EUB_TRUE_1:.+]], label %[[EUB_FALSE_1:.+]]
    // CHECK-DAG: [[EUB_TRUE_1]]:
    // CHECK: [[NUM_IT_3:%.+]] = load{{.+}}
    // CHECK: br label %[[EUB_END_1:.+]]
    // CHECK-DAG: [[EUB_FALSE_1]]:
    // CHECK: [[OMP_UB_VAL3:%.+]] = load{{.+}} [[OMP_UB]],
    // CHECK: br label %[[EUB_END_1]]
    // CHECK-DAG: [[EUB_END_1]]:
    // CHECK-DAG: [[EUB_RES_1:%.+]] = phi{{.+}} [ [[NUM_IT_3]], %[[EUB_TRUE_1]] ], [ [[OMP_UB_VAL3]], %[[EUB_FALSE_1]] ]
    // CHECK: store{{.+}} [[EUB_RES_1]], {{.+}}* [[OMP_UB]],

    // Store LB in IV
    // CHECK-DAG: [[OMP_LB_VAL_3:%.+]] = load{{.+}}, {{.+}} [[OMP_LB]],
    // CHECK: store{{.+}} [[OMP_LB_VAL_3]], {{.+}}* [[OMP_IV]],

    // CHECK: [[DIST_INNER_LOOP_END]]:
    // CHECK: br label %[[LOOP_EXIT:.+]]

    // loop exit
    // CHECK: [[LOOP_EXIT]]:
    // CHECK-DAG: call void @__kmpc_for_static_fini(
    // CHECK: ret

    // skip implementation of 'parallel for': using default scheduling and was tested above
  }

  // schedule: static no chunk
  #pragma omp target
  #pragma omp teams
  // CHECK: define{{.+}} void [[OFFLOADING_FUN_4]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_4:@.+]] to {{.+}})

  #pragma omp distribute parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
    // CHECK: define{{.+}} void [[OMP_OUTLINED_4]](
    // CHECK-DAG: [[OMP_IV:%.omp.iv]] = alloca
    // CHECK-DAG: [[OMP_LB:%.omp.comb.lb]] = alloca
    // CHECK-DAG: [[OMP_UB:%.omp.comb.ub]] = alloca
    // CHECK-DAG: [[OMP_ST:%.omp.stride]] = alloca

    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
    // CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_4:@.+]] to {{.+}},
    // skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
    // CHECK: ret

    // 'parallel for' implementation is the same as the case without schedule clase (static no chunk is the default)
    // CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_4]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})

    // CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},

    // initialize lb and ub to PrevLB and PrevUB
    // CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
    // CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
    // CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
    // CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 34, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})

    // PrevEUB is only used when 'for' has a chunked schedule, otherwise EUB is used
    // In this case we use EUB
    // CHECK-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
    // CHECK: [[PF_NUM_IT_1:%.+]] = load{{.+}},
    // CHECK-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_NUM_IT_1]]
    // CHECK: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
    // CHECK: [[PF_EUB_TRUE]]:
    // CHECK: [[PF_NUM_IT_2:%.+]] = load{{.+}},
    // CHECK: br label %[[PF_EUB_END:.+]]
    // CHECK-DAG: [[PF_EUB_FALSE]]:
    // CHECK: [[OMP_PF_UB_VAL2:%.+]] = load{{.+}} [[OMP_PF_UB]],
    // CHECK: br label %[[PF_EUB_END]]
    // CHECK-DAG: [[PF_EUB_END]]:
    // CHECK-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_NUM_IT_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL2]], %[[PF_EUB_FALSE]] ]
    // CHECK: store{{.+}} [[PF_EUB_RES]],{{.+}}  [[OMP_PF_UB]],

    // initialize omp.iv
    // CHECK: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
    // CHECK: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label %[[OMP_PF_JUMP_BACK:.+]]

    // check exit condition
    // CHECK: [[OMP_PF_JUMP_BACK]]:
    // CHECK-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load {{.+}} [[OMP_PF_IV]],
    // CHECK-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load {{.+}} [[OMP_PF_UB]],
    // CHECK: [[PF_CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
    // CHECK: br {{.+}} [[PF_CMP_IV_UB]], label %[[PF_BODY:.+]], label %[[PF_END:.+]]

    // check that PrevLB and PrevUB are passed to the 'for'
    // CHECK: [[PF_BODY]]:
    // CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label {{.+}}

    // check stride 1 for 'for' in 'distribute parallel for'
    // CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK: [[OMP_PF_IV_INC:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_2]], 1
    // CHECK: store{{.+}} [[OMP_PF_IV_INC]], {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label %[[OMP_PF_JUMP_BACK]]

    // CHECK-DAG: call void @__kmpc_for_static_fini(
    // CHECK: ret
  }

  // schedule: static chunk
  #pragma omp target
  #pragma omp teams
  // CHECK: define{{.+}} void [[OFFLOADING_FUN_5]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}}* [[OMP_OUTLINED_5:@.+]] to {{.+}})

  #pragma omp distribute parallel for schedule(static, ch)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
    // CHECK: define{{.+}} void [[OMP_OUTLINED_5]](
    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
    // CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_5:@.+]] to {{.+}},
    // skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
    // CHECK: ret

    // 'parallel for' implementation using outer and inner loops and PrevEUB
    // CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_5]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}})
    // CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_ST:%.omp.stride]] = alloca{{.+}},

    // initialize lb and ub to PrevLB and PrevUB
    // CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
    // CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
    // CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
    // CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})
    // CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER:.+]]

    // check PrevEUB (using PrevUB instead of NumIt as upper bound)
    // CHECK: [[OMP_PF_OUTER_LOOP_HEADER]]:
    // CHECK-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
    // CHECK-64-DAG: [[OMP_PF_UB_VAL_CONV:%.+]] = sext{{.+}} [[OMP_PF_UB_VAL_1]] to
    // CHECK: [[PF_PREV_UB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
    // CHECK-64-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_CONV]], [[PF_PREV_UB_VAL_1]]
    // CHECK-32-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_PREV_UB_VAL_1]]
    // CHECK: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
    // CHECK: [[PF_EUB_TRUE]]:
    // CHECK: [[PF_PREV_UB_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
    // CHECK: br label %[[PF_EUB_END:.+]]
    // CHECK-DAG: [[PF_EUB_FALSE]]:
    // CHECK: [[OMP_PF_UB_VAL_2:%.+]] = load{{.+}} [[OMP_PF_UB]],
    // CHECK-64: [[OMP_PF_UB_VAL_2_CONV:%.+]] = sext{{.+}} [[OMP_PF_UB_VAL_2]] to
    // CHECK: br label %[[PF_EUB_END]]
    // CHECK-DAG: [[PF_EUB_END]]:
    // CHECK-64-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_PREV_UB_VAL_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL_2_CONV]], %[[PF_EUB_FALSE]] ]
    // CHECK-32-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_PREV_UB_VAL_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL_2]], %[[PF_EUB_FALSE]] ]
    // CHECK-64-DAG: [[PF_EUB_RES_CONV:%.+]] = trunc{{.+}} [[PF_EUB_RES]] to
    // CHECK-64: store{{.+}} [[PF_EUB_RES_CONV]],{{.+}}  [[OMP_PF_UB]],
    // CHECK-32: store{{.+}} [[PF_EUB_RES]], {{.+}} [[OMP_PF_UB]],

    // initialize omp.iv (IV = LB)
    // CHECK: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
    // CHECK: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],

    // outer loop: while (IV < UB) {
    // CHECK-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
    // CHECK: [[PF_CMP_IV_UB_1:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
    // CHECK: br{{.+}} [[PF_CMP_IV_UB_1]], label %[[OMP_PF_OUTER_LOOP_BODY:.+]], label %[[OMP_PF_OUTER_LOOP_END:.+]]

    // CHECK: [[OMP_PF_OUTER_LOOP_BODY]]:
    // CHECK: br label %[[OMP_PF_INNER_FOR_HEADER:.+]]

    // CHECK: [[OMP_PF_INNER_FOR_HEADER]]:
    // CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK-DAG: [[OMP_PF_UB_VAL_4:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
    // CHECK: [[PF_CMP_IV_UB_2:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_2]], [[OMP_PF_UB_VAL_4]]
    // CHECK: br{{.+}} [[PF_CMP_IV_UB_2]], label %[[OMP_PF_INNER_LOOP_BODY:.+]], label %[[OMP_PF_INNER_LOOP_END:.+]]

    // CHECK: [[OMP_PF_INNER_LOOP_BODY]]:
    // CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // skip body branch
    // CHECK: br{{.+}}
    // CHECK: br label %[[OMP_PF_INNER_LOOP_INC:.+]]

    // IV = IV + 1 and inner loop latch
    // CHECK: [[OMP_PF_INNER_LOOP_INC]]:
    // CHECK-DAG: [[OMP_PF_IV_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_IV]],
    // CHECK-DAG: [[OMP_PF_NEXT_IV:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_3]], 1
    // CHECK-DAG: store{{.+}} [[OMP_PF_NEXT_IV]], {{.+}}* [[OMP_IV]],
    // CHECK: br label %[[OMP_PF_INNER_FOR_HEADER]]

    // check NextLB and NextUB
    // CHECK: [[OMP_PF_INNER_LOOP_END]]:
    // CHECK: br label %[[OMP_PF_OUTER_LOOP_INC:.+]]

    // CHECK: [[OMP_PF_OUTER_LOOP_INC]]:
    // CHECK-DAG: [[OMP_PF_LB_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
    // CHECK-DAG: [[OMP_PF_ST_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_ST]],
    // CHECK-DAG: [[OMP_PF_LB_NEXT:%.+]] = add{{.+}} [[OMP_PF_LB_VAL_2]], [[OMP_PF_ST_VAL_1]]
    // CHECK: store{{.+}} [[OMP_PF_LB_NEXT]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-DAG: [[OMP_PF_UB_VAL_5:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_UB]],
    // CHECK-DAG: [[OMP_PF_ST_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_ST]],
    // CHECK-DAG: [[OMP_PF_UB_NEXT:%.+]] = add{{.+}} [[OMP_PF_UB_VAL_5]], [[OMP_PF_ST_VAL_2]]
    // CHECK: store{{.+}} [[OMP_PF_UB_NEXT]], {{.+}}* [[OMP_PF_UB]],
    // CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER]]

    // CHECK: [[OMP_PF_OUTER_LOOP_END]]:
    // CHECK-DAG: call void @__kmpc_for_static_fini(
    // CHECK: ret
  }

  // schedule: dynamic no chunk
  #pragma omp target
  #pragma omp teams
  // CHECK: define{{.+}} void [[OFFLOADING_FUN_6]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_6:@.+]] to {{.+}})

  #pragma omp distribute parallel for schedule(dynamic)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
    // CHECK: define{{.+}} void [[OMP_OUTLINED_6]](
    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
    // CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_6:@.+]] to {{.+}},
    // skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
    // CHECK: ret

    // 'parallel for' implementation using outer and inner loops and PrevEUB
    // CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_6]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})
    // CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_ST:%.omp.stride]] = alloca{{.+}},

    // initialize lb and ub to PrevLB and PrevUB
    // CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
    // CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
    // CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
    // CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
    // CHECK-DAG: [[OMP_PF_LB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
    // CHECK-DAG: [[OMP_PF_UB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_UB]],
    // CHECK: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, {{.+}} 35, {{.+}} [[OMP_PF_LB_VAL]], {{.+}} [[OMP_PF_UB_VAL]], {{.+}}, {{.+}})
    // CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER:.+]]

    // CHECK: [[OMP_PF_OUTER_LOOP_HEADER]]:
    // CHECK: [[IS_FIN:%.+]] = call{{.+}} @__kmpc_dispatch_next_4({{.+}}, {{.+}}, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]], {{.+}}* [[OMP_PF_ST]])
    // CHECK: [[IS_FIN_CMP:%.+]] = icmp{{.+}} [[IS_FIN]], 0
    // CHECK: br{{.+}} [[IS_FIN_CMP]], label %[[OMP_PF_OUTER_LOOP_BODY:.+]], label %[[OMP_PF_OUTER_LOOP_END:.+]]

    // initialize omp.iv (IV = LB)
    // CHECK: [[OMP_PF_OUTER_LOOP_BODY]]:
    // CHECK-DAG: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
    // CHECK-DAG: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label %[[OMP_PF_INNER_LOOP_HEADER:.+]]

    // CHECK: [[OMP_PF_INNER_LOOP_HEADER]]:
    // CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK-DAG: [[OMP_PF_UB_VAL_4:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
    // CHECK: [[PF_CMP_IV_UB_2:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_2]], [[OMP_PF_UB_VAL_4]]
    // CHECK: br{{.+}} [[PF_CMP_IV_UB_2]], label %[[OMP_PF_INNER_LOOP_BODY:.+]], label %[[OMP_PF_INNER_LOOP_END:.+]]

    // CHECK: [[OMP_PF_INNER_LOOP_BODY]]:
    // CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // skip body branch
    // CHECK: br{{.+}}
    // CHECK: br label %[[OMP_PF_INNER_LOOP_INC:.+]]

    // IV = IV + 1 and inner loop latch
    // CHECK: [[OMP_PF_INNER_LOOP_INC]]:
    // CHECK-DAG: [[OMP_PF_IV_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_IV]],
    // CHECK-DAG: [[OMP_PF_NEXT_IV:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_3]], 1
    // CHECK-DAG: store{{.+}} [[OMP_PF_NEXT_IV]], {{.+}}* [[OMP_IV]],
    // CHECK: br label %[[OMP_PF_INNER_LOOP_HEADER]]

    // check NextLB and NextUB
    // CHECK: [[OMP_PF_INNER_LOOP_END]]:
    // CHECK: br label %[[OMP_PF_OUTER_LOOP_INC:.+]]

    // CHECK: [[OMP_PF_OUTER_LOOP_INC]]:
    // CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER]]

    // CHECK: [[OMP_PF_OUTER_LOOP_END]]:
    // CHECK: ret
  }

  // schedule: dynamic chunk
  #pragma omp target
  #pragma omp teams
  // CHECK: define{{.+}} void [[OFFLOADING_FUN_7]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}}* [[OMP_OUTLINED_7:@.+]] to {{.+}})

  #pragma omp distribute parallel for schedule(dynamic, ch)
  for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
    // CHECK: define{{.+}} void [[OMP_OUTLINED_7]](
    // CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
    // CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_7:@.+]] to {{.+}},
    // skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
    // CHECK: ret

    // 'parallel for' implementation using outer and inner loops and PrevEUB
    // CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_7]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}})
    // CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},
    // CHECK-DAG: [[OMP_PF_ST:%.omp.stride]] = alloca{{.+}},

    // initialize lb and ub to PrevLB and PrevUB
    // CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
    // CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
    // CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
    // CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
    // CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
    // CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
    // CHECK-DAG: [[OMP_PF_LB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
    // CHECK-DAG: [[OMP_PF_UB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_UB]],
    // CHECK: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, {{.+}} 35, {{.+}} [[OMP_PF_LB_VAL]], {{.+}} [[OMP_PF_UB_VAL]], {{.+}}, {{.+}})
    // CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER:.+]]

    // CHECK: [[OMP_PF_OUTER_LOOP_HEADER]]:
    // CHECK: [[IS_FIN:%.+]] = call{{.+}} @__kmpc_dispatch_next_4({{.+}}, {{.+}}, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]], {{.+}}* [[OMP_PF_ST]])
    // CHECK: [[IS_FIN_CMP:%.+]] = icmp{{.+}} [[IS_FIN]], 0
    // CHECK: br{{.+}} [[IS_FIN_CMP]], label %[[OMP_PF_OUTER_LOOP_BODY:.+]], label %[[OMP_PF_OUTER_LOOP_END:.+]]

    // initialize omp.iv (IV = LB)
    // CHECK: [[OMP_PF_OUTER_LOOP_BODY]]:
    // CHECK-DAG: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
    // CHECK-DAG: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
    // CHECK: br label %[[OMP_PF_INNER_LOOP_HEADER:.+]]

    // CHECK: [[OMP_PF_INNER_LOOP_HEADER]]:
    // CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // CHECK-DAG: [[OMP_PF_UB_VAL_4:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
    // CHECK: [[PF_CMP_IV_UB_2:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_2]], [[OMP_PF_UB_VAL_4]]
    // CHECK: br{{.+}} [[PF_CMP_IV_UB_2]], label %[[OMP_PF_INNER_LOOP_BODY:.+]], label %[[OMP_PF_INNER_LOOP_END:.+]]

    // CHECK: [[OMP_PF_INNER_LOOP_BODY]]:
    // CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
    // skip body branch
    // CHECK: br{{.+}}
    // CHECK: br label %[[OMP_PF_INNER_LOOP_INC:.+]]

    // IV = IV + 1 and inner loop latch
    // CHECK: [[OMP_PF_INNER_LOOP_INC]]:
    // CHECK-DAG: [[OMP_PF_IV_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_IV]],
    // CHECK-DAG: [[OMP_PF_NEXT_IV:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_3]], 1
    // CHECK-DAG: store{{.+}} [[OMP_PF_NEXT_IV]], {{.+}}* [[OMP_IV]],
    // CHECK: br label %[[OMP_PF_INNER_LOOP_HEADER]]

    // check NextLB and NextUB
    // CHECK: [[OMP_PF_INNER_LOOP_END]]:
    // CHECK: br label %[[OMP_PF_OUTER_LOOP_INC:.+]]

    // CHECK: [[OMP_PF_OUTER_LOOP_INC]]:
    // CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER]]

    // CHECK: [[OMP_PF_OUTER_LOOP_END]]:
    // CHECK: ret
  }

  return tmain<int>();
#endif
}

// check code
// CHECK: define{{.+}} [[TMAIN]]()

// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOADING_FUN_1:@.+]](

// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOADING_FUN_2:@.+]](

// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOADING_FUN_3:@.+]](

// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOADING_FUN_4:@.+]](

// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOADING_FUN_5:@.+]](

// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOADING_FUN_6:@.+]](

// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOADING_FUN_7:@.+]](

// CHECK: define{{.+}} void [[OFFLOADING_FUN_1]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_1:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[OMP_OUTLINED_1]](
// CHECK-DAG: [[OMP_IV:%.omp.iv]] = alloca
// CHECK-DAG: [[OMP_LB:%.omp.comb.lb]] = alloca
// CHECK-DAG: [[OMP_UB:%.omp.comb.ub]] = alloca
// CHECK-DAG: [[OMP_ST:%.omp.stride]] = alloca

// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,

// check EUB for distribute
// CHECK-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK: [[NUM_IT_1:%.+]] = load{{.+}},
// CHECK-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], [[NUM_IT_1]]
// CHECK: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
// CHECK-DAG: [[EUB_TRUE]]:
// CHECK: [[NUM_IT_2:%.+]] = load{{.+}},
// CHECK: br label %[[EUB_END:.+]]
// CHECK-DAG: [[EUB_FALSE]]:
// CHECK: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK: br label %[[EUB_END]]
// CHECK-DAG: [[EUB_END]]:
// CHECK-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ [[NUM_IT_2]], %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
// CHECK: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

// initialize omp.iv
// CHECK: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
// CHECK: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],
// CHECK: br label %[[OMP_JUMP_BACK:.+]]

// check exit condition
// CHECK: [[OMP_JUMP_BACK]]:
// CHECK-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
// CHECK-DAG: [[OMP_UB_VAL_3:%.+]] = load {{.+}} [[OMP_UB]],
// CHECK: [[CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_IV_VAL_1]], [[OMP_UB_VAL_3]]
// CHECK: br {{.+}} [[CMP_IV_UB]], label %[[DIST_BODY:.+]], label %[[DIST_END:.+]]

// check that PrevLB and PrevUB are passed to the 'for'
// CHECK: [[DIST_BODY]]:
// CHECK-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
// CHECK-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
// CHECK-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to {{.+}}
// check that distlb and distub are properly passed to fork_call
// CHECK-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_1:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_EXT]], i{{[0-9]+}} [[OMP_PREV_UB_EXT]], {{.+}})
// CHECK-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_1:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB]], i{{[0-9]+}} [[OMP_PREV_UB]], {{.+}})
// CHECK: br label %[[DIST_INC:.+]]

// increment by stride (distInc - 'parallel for' executes the whole chunk) and latch
// CHECK: [[DIST_INC]]:
// CHECK-DAG: [[OMP_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
// CHECK-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
// CHECK: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_2]], [[OMP_ST_VAL_1]]
// CHECK: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
// CHECK: br label %[[OMP_JUMP_BACK]]

// CHECK-DAG: call void @__kmpc_for_static_fini(
// CHECK: ret

// implementation of 'parallel for'
// CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_1]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})

// CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},

// initialize lb and ub to PrevLB and PrevUB
// CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
// CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
// CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 34, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})

// PrevEUB is only used when 'for' has a chunked schedule, otherwise EUB is used
// In this case we use EUB
// CHECK-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
// CHECK: [[PF_NUM_IT_1:%.+]] = load{{.+}},
// CHECK-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_NUM_IT_1]]
// CHECK: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
// CHECK: [[PF_EUB_TRUE]]:
// CHECK: [[PF_NUM_IT_2:%.+]] = load{{.+}},
// CHECK: br label %[[PF_EUB_END:.+]]
// CHECK-DAG: [[PF_EUB_FALSE]]:
// CHECK: [[OMP_PF_UB_VAL2:%.+]] = load{{.+}} [[OMP_PF_UB]],
// CHECK: br label %[[PF_EUB_END]]
// CHECK-DAG: [[PF_EUB_END]]:
// CHECK-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_NUM_IT_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL2]], %[[PF_EUB_FALSE]] ]
// CHECK: store{{.+}} [[PF_EUB_RES]],{{.+}}  [[OMP_PF_UB]],

// initialize omp.iv
// CHECK: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
// CHECK: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
// CHECK: br label %[[OMP_PF_JUMP_BACK:.+]]

// check exit condition
// CHECK: [[OMP_PF_JUMP_BACK]]:
// CHECK-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load {{.+}} [[OMP_PF_IV]],
// CHECK-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load {{.+}} [[OMP_PF_UB]],
// CHECK: [[PF_CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
// CHECK: br {{.+}} [[PF_CMP_IV_UB]], label %[[PF_BODY:.+]], label %[[PF_END:.+]]

// check that PrevLB and PrevUB are passed to the 'for'
// CHECK: [[PF_BODY]]:
// CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK: br label {{.+}}

// check stride 1 for 'for' in 'distribute parallel for'
// CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK: [[OMP_PF_IV_INC:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_2]], 1
// CHECK: store{{.+}} [[OMP_PF_IV_INC]], {{.+}}* [[OMP_PF_IV]],
// CHECK: br label %[[OMP_PF_JUMP_BACK]]

// CHECK-DAG: call void @__kmpc_for_static_fini(
// CHECK: ret

// CHECK: define{{.+}} void [[OFFLOADING_FUN_2]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_2:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[OMP_OUTLINED_2]](
// CHECK-DAG: [[OMP_IV:%.omp.iv]] = alloca
// CHECK-DAG: [[OMP_LB:%.omp.comb.lb]] = alloca
// CHECK-DAG: [[OMP_UB:%.omp.comb.ub]] = alloca
// CHECK-DAG: [[OMP_ST:%.omp.stride]] = alloca

// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,

// check EUB for distribute
// CHECK-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK: [[NUM_IT_1:%.+]] = load{{.+}},
// CHECK-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], [[NUM_IT_1]]
// CHECK: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
// CHECK-DAG: [[EUB_TRUE]]:
// CHECK: [[NUM_IT_2:%.+]] = load{{.+}},
// CHECK: br label %[[EUB_END:.+]]
// CHECK-DAG: [[EUB_FALSE]]:
// CHECK: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK: br label %[[EUB_END]]
// CHECK-DAG: [[EUB_END]]:
// CHECK-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ [[NUM_IT_2]], %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
// CHECK: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

// initialize omp.iv
// CHECK: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
// CHECK: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],
// CHECK: br label %[[OMP_JUMP_BACK:.+]]

// check exit condition
// CHECK: [[OMP_JUMP_BACK]]:
// CHECK-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
// CHECK-DAG: [[OMP_UB_VAL_3:%.+]] = load {{.+}} [[OMP_UB]],
// CHECK: [[CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_IV_VAL_1]], [[OMP_UB_VAL_3]]
// CHECK: br {{.+}} [[CMP_IV_UB]], label %[[DIST_BODY:.+]], label %[[DIST_END:.+]]

// check that PrevLB and PrevUB are passed to the 'for'
// CHECK: [[DIST_BODY]]:
// CHECK-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
// CHECK-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
// CHECK-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to {{.+}}
// check that distlb and distub are properly passed to fork_call
// CHECK-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_2:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_EXT]], i{{[0-9]+}} [[OMP_PREV_UB_EXT]], {{.+}})
// CHECK-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_2:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB]], i{{[0-9]+}} [[OMP_PREV_UB]], {{.+}})
// CHECK: br label %[[DIST_INC:.+]]

// increment by stride (distInc - 'parallel for' executes the whole chunk) and latch
// CHECK: [[DIST_INC]]:
// CHECK-DAG: [[OMP_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
// CHECK-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
// CHECK: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_2]], [[OMP_ST_VAL_1]]
// CHECK: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
// CHECK: br label %[[OMP_JUMP_BACK]]

// CHECK-DAG: call void @__kmpc_for_static_fini(
// CHECK: ret

// implementation of 'parallel for'
// CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_2]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})

// CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},

// initialize lb and ub to PrevLB and PrevUB
// CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
// CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
// CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 34, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})

// PrevEUB is only used when 'for' has a chunked schedule, otherwise EUB is used
// In this case we use EUB
// CHECK-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
// CHECK: [[PF_NUM_IT_1:%.+]] = load{{.+}},
// CHECK-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_NUM_IT_1]]
// CHECK: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
// CHECK: [[PF_EUB_TRUE]]:
// CHECK: [[PF_NUM_IT_2:%.+]] = load{{.+}},
// CHECK: br label %[[PF_EUB_END:.+]]
// CHECK-DAG: [[PF_EUB_FALSE]]:
// CHECK: [[OMP_PF_UB_VAL2:%.+]] = load{{.+}} [[OMP_PF_UB]],
// CHECK: br label %[[PF_EUB_END]]
// CHECK-DAG: [[PF_EUB_END]]:
// CHECK-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_NUM_IT_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL2]], %[[PF_EUB_FALSE]] ]
// CHECK: store{{.+}} [[PF_EUB_RES]],{{.+}}  [[OMP_PF_UB]],

// initialize omp.iv
// CHECK: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
// CHECK: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
// CHECK: br label %[[OMP_PF_JUMP_BACK:.+]]

// check exit condition
// CHECK: [[OMP_PF_JUMP_BACK]]:
// CHECK-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load {{.+}} [[OMP_PF_IV]],
// CHECK-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load {{.+}} [[OMP_PF_UB]],
// CHECK: [[PF_CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
// CHECK: br {{.+}} [[PF_CMP_IV_UB]], label %[[PF_BODY:.+]], label %[[PF_END:.+]]

// check that PrevLB and PrevUB are passed to the 'for'
// CHECK: [[PF_BODY]]:
// CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK: br label {{.+}}

// check stride 1 for 'for' in 'distribute parallel for'
// CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK: [[OMP_PF_IV_INC:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_2]], 1
// CHECK: store{{.+}} [[OMP_PF_IV_INC]], {{.+}}* [[OMP_PF_IV]],
// CHECK: br label %[[OMP_PF_JUMP_BACK]]

// CHECK-DAG: call void @__kmpc_for_static_fini(
// CHECK: ret

// CHECK: define{{.+}} void [[OFFLOADING_FUN_3]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}}* [[OMP_OUTLINED_3:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[OMP_OUTLINED_3]](
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: [[OMP_IV:%.+]] = alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: [[OMP_LB:%.+]] = alloca
// CHECK: [[OMP_UB:%.+]] = alloca
// CHECK: [[OMP_ST:%.+]] = alloca

// unlike the previous tests, in this one we have a outer and inner loop for 'distribute'
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 91,

// check EUB for distribute
// CHECK-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK: [[NUM_IT_1:%.+]] = load{{.+}}
// CHECK-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], [[NUM_IT_1]]
// CHECK: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
// CHECK-DAG: [[EUB_TRUE]]:
// CHECK: [[NUM_IT_2:%.+]] = load{{.+}},
// CHECK: br label %[[EUB_END:.+]]
// CHECK-DAG: [[EUB_FALSE]]:
// CHECK: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK: br label %[[EUB_END]]
// CHECK-DAG: [[EUB_END]]:
// CHECK-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ [[NUM_IT_2]], %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
// CHECK: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

// initialize omp.iv
// CHECK: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
// CHECK: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],

// check exit condition
// CHECK-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
// CHECK-DAG: [[OMP_UB_VAL_3:%.+]] = load {{.+}}
// CHECK-DAG: [[OMP_UB_VAL_3_PLUS_ONE:%.+]] = add {{.+}} [[OMP_UB_VAL_3]], 1
// CHECK: [[CMP_IV_UB:%.+]] = icmp slt {{.+}} [[OMP_IV_VAL_1]], [[OMP_UB_VAL_3_PLUS_ONE]]
// CHECK: br {{.+}} [[CMP_IV_UB]], label %[[DIST_INNER_LOOP_BODY:.+]], label %[[DIST_INNER_LOOP_END:.+]]

// check that PrevLB and PrevUB are passed to the 'for'
// CHECK: [[DIST_INNER_LOOP_BODY]]:
// CHECK-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
// CHECK-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
// CHECK-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to {{.+}}
// check that distlb and distub are properly passed to fork_call
// CHECK-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_3:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_EXT]], i{{[0-9]+}} [[OMP_PREV_UB_EXT]], {{.+}})
// CHECK-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_3:@.+]] to {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB]], i{{[0-9]+}} [[OMP_PREV_UB]], {{.+}})
// CHECK: br label %[[DIST_INNER_LOOP_INC:.+]]

// check DistInc
// CHECK: [[DIST_INNER_LOOP_INC]]:
// CHECK-DAG: [[OMP_IV_VAL_3:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
// CHECK-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
// CHECK: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_3]], [[OMP_ST_VAL_1]]
// CHECK: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
// CHECK-DAG: [[OMP_LB_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_LB]],
// CHECK-DAG: [[OMP_ST_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_ST]],
// CHECK-DAG: [[OMP_LB_NEXT:%.+]] = add{{.+}} [[OMP_LB_VAL_2]], [[OMP_ST_VAL_2]]
// CHECK: store{{.+}} [[OMP_LB_NEXT]], {{.+}}* [[OMP_LB]],
// CHECK-DAG: [[OMP_UB_VAL_5:%.+]] = load{{.+}}, {{.+}} [[OMP_UB]],
// CHECK-DAG: [[OMP_ST_VAL_3:%.+]] = load{{.+}}, {{.+}} [[OMP_ST]],
// CHECK-DAG: [[OMP_UB_NEXT:%.+]] = add{{.+}} [[OMP_UB_VAL_5]], [[OMP_ST_VAL_3]]
// CHECK: store{{.+}} [[OMP_UB_NEXT]], {{.+}}* [[OMP_UB]],

// Update UB
// CHECK-DAG: [[OMP_UB_VAL_6:%.+]] = load{{.+}}, {{.+}} [[OMP_UB]],
// CHECK: [[OMP_EXPR_VAL:%.+]] = load{{.+}}, {{.+}}
// CHECK-DAG: [[CMP_UB_NUM_IT_1:%.+]] = icmp sgt {{.+}}[[OMP_UB_VAL_6]], [[OMP_EXPR_VAL]]
// CHECK: br {{.+}} [[CMP_UB_NUM_IT_1]], label %[[EUB_TRUE_1:.+]], label %[[EUB_FALSE_1:.+]]
// CHECK-DAG: [[EUB_TRUE_1]]:
// CHECK: [[NUM_IT_3:%.+]] = load{{.+}}
// CHECK: br label %[[EUB_END_1:.+]]
// CHECK-DAG: [[EUB_FALSE_1]]:
// CHECK: [[OMP_UB_VAL3:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK: br label %[[EUB_END_1]]
// CHECK-DAG: [[EUB_END_1]]:
// CHECK-DAG: [[EUB_RES_1:%.+]] = phi{{.+}} [ [[NUM_IT_3]], %[[EUB_TRUE_1]] ], [ [[OMP_UB_VAL3]], %[[EUB_FALSE_1]] ]
// CHECK: store{{.+}} [[EUB_RES_1]], {{.+}}* [[OMP_UB]],

// Store LB in IV
// CHECK-DAG: [[OMP_LB_VAL_3:%.+]] = load{{.+}}, {{.+}} [[OMP_LB]],
// CHECK: store{{.+}} [[OMP_LB_VAL_3]], {{.+}}* [[OMP_IV]],

// CHECK: [[DIST_INNER_LOOP_END]]:
// CHECK: br label %[[LOOP_EXIT:.+]]

// loop exit
// CHECK: [[LOOP_EXIT]]:
// CHECK-DAG: call void @__kmpc_for_static_fini(
// CHECK: ret

// skip implementation of 'parallel for': using default scheduling and was tested above

// CHECK: define{{.+}} void [[OFFLOADING_FUN_4]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_4:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[OMP_OUTLINED_4]](
// CHECK-DAG: [[OMP_IV:%.omp.iv]] = alloca
// CHECK-DAG: [[OMP_LB:%.omp.comb.lb]] = alloca
// CHECK-DAG: [[OMP_UB:%.omp.comb.ub]] = alloca
// CHECK-DAG: [[OMP_ST:%.omp.stride]] = alloca

// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
// CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_4:@.+]] to {{.+}},
// skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
// CHECK: ret

// 'parallel for' implementation is the same as the case without schedule clase (static no chunk is the default)
// CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_4]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})

// CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},

// initialize lb and ub to PrevLB and PrevUB
// CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
// CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
// CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 34, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})

// PrevEUB is only used when 'for' has a chunked schedule, otherwise EUB is used
// In this case we use EUB
// CHECK-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
// CHECK: [[PF_NUM_IT_1:%.+]] = load{{.+}},
// CHECK-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_NUM_IT_1]]
// CHECK: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
// CHECK: [[PF_EUB_TRUE]]:
// CHECK: [[PF_NUM_IT_2:%.+]] = load{{.+}},
// CHECK: br label %[[PF_EUB_END:.+]]
// CHECK-DAG: [[PF_EUB_FALSE]]:
// CHECK: [[OMP_PF_UB_VAL2:%.+]] = load{{.+}} [[OMP_PF_UB]],
// CHECK: br label %[[PF_EUB_END]]
// CHECK-DAG: [[PF_EUB_END]]:
// CHECK-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_NUM_IT_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL2]], %[[PF_EUB_FALSE]] ]
// CHECK: store{{.+}} [[PF_EUB_RES]],{{.+}}  [[OMP_PF_UB]],

// initialize omp.iv
// CHECK: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
// CHECK: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
// CHECK: br label %[[OMP_PF_JUMP_BACK:.+]]

// check exit condition
// CHECK: [[OMP_PF_JUMP_BACK]]:
// CHECK-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load {{.+}} [[OMP_PF_IV]],
// CHECK-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load {{.+}} [[OMP_PF_UB]],
// CHECK: [[PF_CMP_IV_UB:%.+]] = icmp sle {{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
// CHECK: br {{.+}} [[PF_CMP_IV_UB]], label %[[PF_BODY:.+]], label %[[PF_END:.+]]

// check that PrevLB and PrevUB are passed to the 'for'
// CHECK: [[PF_BODY]]:
// CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK: br label {{.+}}

// check stride 1 for 'for' in 'distribute parallel for'
// CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load {{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK: [[OMP_PF_IV_INC:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_2]], 1
// CHECK: store{{.+}} [[OMP_PF_IV_INC]], {{.+}}* [[OMP_PF_IV]],
// CHECK: br label %[[OMP_PF_JUMP_BACK]]

// CHECK-DAG: call void @__kmpc_for_static_fini(
// CHECK: ret

// CHECK: define{{.+}} void [[OFFLOADING_FUN_5]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}}* [[OMP_OUTLINED_5:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[OMP_OUTLINED_5]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
// CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_5:@.+]] to {{.+}},
// skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
// CHECK: ret

// 'parallel for' implementation using outer and inner loops and PrevEUB
// CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_5]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}})
// CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_ST:%.omp.stride]] = alloca{{.+}},

// initialize lb and ub to PrevLB and PrevUB
// CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
// CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
// CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]],{{.+}})
// CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER:.+]]

// check PrevEUB (using PrevUB instead of NumIt as upper bound)
// CHECK: [[OMP_PF_OUTER_LOOP_HEADER]]:
// CHECK-DAG: [[OMP_PF_UB_VAL_1:%.+]] = load{{.+}} [[OMP_PF_UB]],
// CHECK-64-DAG: [[OMP_PF_UB_VAL_CONV:%.+]] = sext{{.+}} [[OMP_PF_UB_VAL_1]] to
// CHECK: [[PF_PREV_UB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
// CHECK-64-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_CONV]], [[PF_PREV_UB_VAL_1]]
// CHECK-32-DAG: [[PF_CMP_UB_NUM_IT:%.+]] = icmp{{.+}} [[OMP_PF_UB_VAL_1]], [[PF_PREV_UB_VAL_1]]
// CHECK: br i1 [[PF_CMP_UB_NUM_IT]], label %[[PF_EUB_TRUE:.+]], label %[[PF_EUB_FALSE:.+]]
// CHECK: [[PF_EUB_TRUE]]:
// CHECK: [[PF_PREV_UB_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
// CHECK: br label %[[PF_EUB_END:.+]]
// CHECK-DAG: [[PF_EUB_FALSE]]:
// CHECK: [[OMP_PF_UB_VAL_2:%.+]] = load{{.+}} [[OMP_PF_UB]],
// CHECK-64: [[OMP_PF_UB_VAL_2_CONV:%.+]] = sext{{.+}} [[OMP_PF_UB_VAL_2]] to
// CHECK: br label %[[PF_EUB_END]]
// CHECK-DAG: [[PF_EUB_END]]:
// CHECK-64-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_PREV_UB_VAL_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL_2_CONV]], %[[PF_EUB_FALSE]] ]
// CHECK-32-DAG: [[PF_EUB_RES:%.+]] = phi{{.+}} [ [[PF_PREV_UB_VAL_2]], %[[PF_EUB_TRUE]] ], [ [[OMP_PF_UB_VAL_2]], %[[PF_EUB_FALSE]] ]
// CHECK-64-DAG: [[PF_EUB_RES_CONV:%.+]] = trunc{{.+}} [[PF_EUB_RES]] to
// CHECK-64: store{{.+}} [[PF_EUB_RES_CONV]],{{.+}}  [[OMP_PF_UB]],
// CHECK-32: store{{.+}} [[PF_EUB_RES]], {{.+}} [[OMP_PF_UB]],

// initialize omp.iv (IV = LB)
// CHECK: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
// CHECK: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],

// outer loop: while (IV < UB) {
// CHECK-DAG: [[OMP_PF_IV_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK-DAG: [[OMP_PF_UB_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
// CHECK: [[PF_CMP_IV_UB_1:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_1]], [[OMP_PF_UB_VAL_3]]
// CHECK: br{{.+}} [[PF_CMP_IV_UB_1]], label %[[OMP_PF_OUTER_LOOP_BODY:.+]], label %[[OMP_PF_OUTER_LOOP_END:.+]]

// CHECK: [[OMP_PF_OUTER_LOOP_BODY]]:
// CHECK: br label %[[OMP_PF_INNER_FOR_HEADER:.+]]

// CHECK: [[OMP_PF_INNER_FOR_HEADER]]:
// CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK-DAG: [[OMP_PF_UB_VAL_4:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
// CHECK: [[PF_CMP_IV_UB_2:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_2]], [[OMP_PF_UB_VAL_4]]
// CHECK: br{{.+}} [[PF_CMP_IV_UB_2]], label %[[OMP_PF_INNER_LOOP_BODY:.+]], label %[[OMP_PF_INNER_LOOP_END:.+]]

// CHECK: [[OMP_PF_INNER_LOOP_BODY]]:
// CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// skip body branch
// CHECK: br{{.+}}
// CHECK: br label %[[OMP_PF_INNER_LOOP_INC:.+]]

// IV = IV + 1 and inner loop latch
// CHECK: [[OMP_PF_INNER_LOOP_INC]]:
// CHECK-DAG: [[OMP_PF_IV_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_IV]],
// CHECK-DAG: [[OMP_PF_NEXT_IV:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_3]], 1
// CHECK-DAG: store{{.+}} [[OMP_PF_NEXT_IV]], {{.+}}* [[OMP_IV]],
// CHECK: br label %[[OMP_PF_INNER_FOR_HEADER]]

// check NextLB and NextUB
// CHECK: [[OMP_PF_INNER_LOOP_END]]:
// CHECK: br label %[[OMP_PF_OUTER_LOOP_INC:.+]]

// CHECK: [[OMP_PF_OUTER_LOOP_INC]]:
// CHECK-DAG: [[OMP_PF_LB_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
// CHECK-DAG: [[OMP_PF_ST_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_ST]],
// CHECK-DAG: [[OMP_PF_LB_NEXT:%.+]] = add{{.+}} [[OMP_PF_LB_VAL_2]], [[OMP_PF_ST_VAL_1]]
// CHECK: store{{.+}} [[OMP_PF_LB_NEXT]], {{.+}}* [[OMP_PF_LB]],
// CHECK-DAG: [[OMP_PF_UB_VAL_5:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_UB]],
// CHECK-DAG: [[OMP_PF_ST_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_ST]],
// CHECK-DAG: [[OMP_PF_UB_NEXT:%.+]] = add{{.+}} [[OMP_PF_UB_VAL_5]], [[OMP_PF_ST_VAL_2]]
// CHECK: store{{.+}} [[OMP_PF_UB_NEXT]], {{.+}}* [[OMP_PF_UB]],
// CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER]]

// CHECK: [[OMP_PF_OUTER_LOOP_END]]:
// CHECK-DAG: call void @__kmpc_for_static_fini(
// CHECK: ret

// CHECK: define{{.+}} void [[OFFLOADING_FUN_6]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED_6:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[OMP_OUTLINED_6]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
// CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_6:@.+]] to {{.+}},
// skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
// CHECK: ret

// 'parallel for' implementation using outer and inner loops and PrevEUB
// CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_6]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}})
// CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_ST:%.omp.stride]] = alloca{{.+}},

// initialize lb and ub to PrevLB and PrevUB
// CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
// CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
// CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
// CHECK-DAG: [[OMP_PF_LB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
// CHECK-DAG: [[OMP_PF_UB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_UB]],
// CHECK: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, {{.+}} 35, {{.+}} [[OMP_PF_LB_VAL]], {{.+}} [[OMP_PF_UB_VAL]], {{.+}}, {{.+}})
// CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER:.+]]

// CHECK: [[OMP_PF_OUTER_LOOP_HEADER]]:
// CHECK: [[IS_FIN:%.+]] = call{{.+}} @__kmpc_dispatch_next_4({{.+}}, {{.+}}, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]], {{.+}}* [[OMP_PF_ST]])
// CHECK: [[IS_FIN_CMP:%.+]] = icmp{{.+}} [[IS_FIN]], 0
// CHECK: br{{.+}} [[IS_FIN_CMP]], label %[[OMP_PF_OUTER_LOOP_BODY:.+]], label %[[OMP_PF_OUTER_LOOP_END:.+]]

// initialize omp.iv (IV = LB)
// CHECK: [[OMP_PF_OUTER_LOOP_BODY]]:
// CHECK-DAG: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
// CHECK-DAG: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
// CHECK: br label %[[OMP_PF_INNER_LOOP_HEADER:.+]]

// CHECK: [[OMP_PF_INNER_LOOP_HEADER]]:
// CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK-DAG: [[OMP_PF_UB_VAL_4:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
// CHECK: [[PF_CMP_IV_UB_2:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_2]], [[OMP_PF_UB_VAL_4]]
// CHECK: br{{.+}} [[PF_CMP_IV_UB_2]], label %[[OMP_PF_INNER_LOOP_BODY:.+]], label %[[OMP_PF_INNER_LOOP_END:.+]]

// CHECK: [[OMP_PF_INNER_LOOP_BODY]]:
// CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// skip body branch
// CHECK: br{{.+}}
// CHECK: br label %[[OMP_PF_INNER_LOOP_INC:.+]]

// IV = IV + 1 and inner loop latch
// CHECK: [[OMP_PF_INNER_LOOP_INC]]:
// CHECK-DAG: [[OMP_PF_IV_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_IV]],
// CHECK-DAG: [[OMP_PF_NEXT_IV:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_3]], 1
// CHECK-DAG: store{{.+}} [[OMP_PF_NEXT_IV]], {{.+}}* [[OMP_IV]],
// CHECK: br label %[[OMP_PF_INNER_LOOP_HEADER]]

// check NextLB and NextUB
// CHECK: [[OMP_PF_INNER_LOOP_END]]:
// CHECK: br label %[[OMP_PF_OUTER_LOOP_INC:.+]]

// CHECK: [[OMP_PF_OUTER_LOOP_INC]]:
// CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER]]

// CHECK: [[OMP_PF_OUTER_LOOP_END]]:
// CHECK: ret

// CHECK: define{{.+}} void [[OFFLOADING_FUN_7]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}}* [[OMP_OUTLINED_7:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[OMP_OUTLINED_7]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92,
// CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_7:@.+]] to {{.+}},
// skip rest of implementation of 'distribute' as it is tested above for default dist_schedule case
// CHECK: ret

// 'parallel for' implementation using outer and inner loops and PrevEUB
// CHECK: define{{.+}} void [[OMP_PARFOR_OUTLINED_7]]({{.+}}, {{.+}}, i{{[0-9]+}} [[OMP_PREV_LB_IN:%.+]], i{{[0-9]+}} [[OMP_PREV_UB_IN:%.+]], {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}})
// CHECK-DAG: [[OMP_PF_LB:%.omp.lb]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_UB:%.omp.ub]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_IV:%.omp.iv]] = alloca{{.+}},
// CHECK-DAG: [[OMP_PF_ST:%.omp.stride]] = alloca{{.+}},

// initialize lb and ub to PrevLB and PrevUB
// CHECK-DAG: store{{.+}} [[OMP_PREV_LB_IN]], {{.+}}* [[PREV_LB_ADDR:%.+]],
// CHECK-DAG: store{{.+}} [[OMP_PREV_UB_IN]], {{.+}}* [[PREV_UB_ADDR:%.+]],
// CHECK-DAG: [[OMP_PREV_LB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_LB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_LB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_LB_VAL]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB_VAL:%.+]] = load{{.+}}, {{.+}}* [[PREV_UB_ADDR]],
// CHECK-64-DAG: [[OMP_PREV_UB_TRC:%.+]] = trunc{{.+}} [[OMP_PREV_UB_VAL]] to {{.+}}
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_LB_TRC]], {{.+}}* [[OMP_PF_LB]],
// CHECK-64-DAG: store{{.+}} [[OMP_PREV_UB_TRC]], {{.+}}* [[OMP_PF_UB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_LB_VAL]], {{.+}}* [[OMP_PF_LB]],
// CHECK-32-DAG: store{{.+}} [[OMP_PREV_UB_VAL]], {{.+}}* [[OMP_PF_UB]],
// CHECK-DAG: [[OMP_PF_LB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
// CHECK-DAG: [[OMP_PF_UB_VAL:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_UB]],
// CHECK: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, {{.+}} 35, {{.+}} [[OMP_PF_LB_VAL]], {{.+}} [[OMP_PF_UB_VAL]], {{.+}}, {{.+}})
// CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER:.+]]

// CHECK: [[OMP_PF_OUTER_LOOP_HEADER]]:
// CHECK: [[IS_FIN:%.+]] = call{{.+}} @__kmpc_dispatch_next_4({{.+}}, {{.+}}, {{.+}}, {{.+}}* [[OMP_PF_LB]], {{.+}}* [[OMP_PF_UB]], {{.+}}* [[OMP_PF_ST]])
// CHECK: [[IS_FIN_CMP:%.+]] = icmp{{.+}} [[IS_FIN]], 0
// CHECK: br{{.+}} [[IS_FIN_CMP]], label %[[OMP_PF_OUTER_LOOP_BODY:.+]], label %[[OMP_PF_OUTER_LOOP_END:.+]]

// initialize omp.iv (IV = LB)
// CHECK: [[OMP_PF_OUTER_LOOP_BODY]]:
// CHECK-DAG: [[OMP_PF_LB_VAL_1:%.+]] = load{{.+}}, {{.+}} [[OMP_PF_LB]],
// CHECK-DAG: store {{.+}} [[OMP_PF_LB_VAL_1]], {{.+}}* [[OMP_PF_IV]],
// CHECK: br label %[[OMP_PF_INNER_LOOP_HEADER:.+]]

// CHECK: [[OMP_PF_INNER_LOOP_HEADER]]:
// CHECK-DAG: [[OMP_PF_IV_VAL_2:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// CHECK-DAG: [[OMP_PF_UB_VAL_4:%.+]] = load{{.+}}, {{.+}}* [[OMP_PF_UB]],
// CHECK: [[PF_CMP_IV_UB_2:%.+]] = icmp{{.+}} [[OMP_PF_IV_VAL_2]], [[OMP_PF_UB_VAL_4]]
// CHECK: br{{.+}} [[PF_CMP_IV_UB_2]], label %[[OMP_PF_INNER_LOOP_BODY:.+]], label %[[OMP_PF_INNER_LOOP_END:.+]]

// CHECK: [[OMP_PF_INNER_LOOP_BODY]]:
// CHECK-DAG: {{.+}} = load{{.+}}, {{.+}}* [[OMP_PF_IV]],
// skip body branch
// CHECK: br{{.+}}
// CHECK: br label %[[OMP_PF_INNER_LOOP_INC:.+]]

// IV = IV + 1 and inner loop latch
// CHECK: [[OMP_PF_INNER_LOOP_INC]]:
// CHECK-DAG: [[OMP_PF_IV_VAL_3:%.+]] = load{{.+}}, {{.+}}* [[OMP_IV]],
// CHECK-DAG: [[OMP_PF_NEXT_IV:%.+]] = add{{.+}} [[OMP_PF_IV_VAL_3]], 1
// CHECK-DAG: store{{.+}} [[OMP_PF_NEXT_IV]], {{.+}}* [[OMP_IV]],
// CHECK: br label %[[OMP_PF_INNER_LOOP_HEADER]]

// check NextLB and NextUB
// CHECK: [[OMP_PF_INNER_LOOP_END]]:
// CHECK: br label %[[OMP_PF_OUTER_LOOP_INC:.+]]

// CHECK: [[OMP_PF_OUTER_LOOP_INC]]:
// CHECK: br label %[[OMP_PF_OUTER_LOOP_HEADER]]

// CHECK: [[OMP_PF_OUTER_LOOP_END]]:
// CHECK: ret
#endif
