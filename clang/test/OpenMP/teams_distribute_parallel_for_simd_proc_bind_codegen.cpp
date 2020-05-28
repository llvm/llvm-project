// add -fopenmp-targets

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

typedef __INTPTR_TYPE__ intptr_t;

// CHECK-DAG: [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DAG: [[DEF_LOC_2:@.+]] = private unnamed_addr constant [[IDENT_T_TY]] { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

void foo();

struct S {
  intptr_t a, b, c;
  S(intptr_t a) : a(a) {}
  operator char() { return a; }
  ~S() {}
};

template <typename T>
T tmain() {
#pragma omp target
#pragma omp teams distribute parallel for simd proc_bind(master)
  for(int i = 0; i < 1000; i++) {}
  return T();
}

int main() {
  // CHECK-LABEL: @main
#pragma omp target
#pragma omp teams distribute parallel for simd proc_bind(spread)
  for(int i = 0; i < 1000; i++) {}
#pragma omp target
#pragma omp teams distribute parallel for simd proc_bind(close)
  for(int i = 0; i < 1000; i++) {}
  return tmain<int>();
}

// CHECK: call {{.*}}@__tgt_target_teams_mapper({{.+}})
// CHECK: call void [[OFFL1:@.+]]()
// CHECK: call {{.*}}@__tgt_target_teams_mapper({{.+}})
// CHECK: call void [[OFFL2:@.+]]()
// CHECK: [[CALL_RET:%.+]] = call{{.+}} i32 [[TMAIN:@.+]]()
// CHECK: ret i32 [[CALL_RET]]

// CHECK: define{{.+}} void [[OFFL1]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, {{.+}}, {{.+}}* [[OMP_OUTLINED_1:@.+]] to {{.+}})

// CHECK: define{{.+}} [[OMP_OUTLINED_1]](i32* {{.+}} [[GTID_IN:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = alloca i32*,
// CHECK: store i32* [[GTID_IN]], i32** [[GTID_ADDR]],
// CHECK: [[GTID_REF:%.+]] = load i32*, i32** [[GTID_ADDR]],
// CHECK: [[GTID_VAL:%.+]] = load i32, i32* [[GTID_REF]],
// CHECK: call {{.*}}void @__kmpc_push_proc_bind([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID_VAL]], i32 4)
// CHECK: call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: ret void

// CHECK: define{{.+}} [[OFFL2]]()
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, {{.+}}, {{.+}}* [[OMP_OUTLINED_1:@.+]] to {{.+}})

// CHECK: define{{.+}} [[OMP_OUTLINED_1]](i32* {{.+}} [[GTID_IN:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = alloca i32*,
// CHECK: store i32* [[GTID_IN]], i32** [[GTID_ADDR]],
// CHECK: [[GTID_REF:%.+]] = load i32*, i32** [[GTID_ADDR]],
// CHECK: [[GTID_VAL:%.+]] = load i32, i32* [[GTID_REF]],
// CHECK: call {{.*}}void @__kmpc_push_proc_bind([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID_VAL]], i32 3)
// CHECK: call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: ret void

// CHECK: define{{.+}} [[TMAIN]]()
// CHECK: call {{.*}}@__tgt_target_teams_mapper({{.+}})
// CHECK: call void [[OFFL3:@.+]]()

// CHECK: define{{.+}} [[OFFL3]]()
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, {{.+}}, {{.+}}* [[OMP_OUTLINED_3:@.+]] to {{.+}})

// CHECK: define{{.+}} [[OMP_OUTLINED_3]](i32* {{.+}} [[GTID_IN:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = alloca i32*,
// CHECK: store i32* [[GTID_IN]], i32** [[GTID_ADDR]],
// CHECK: [[GTID_REF:%.+]] = load i32*, i32** [[GTID_ADDR]],
// CHECK: [[GTID_VAL:%.+]] = load i32, i32* [[GTID_REF]],
// CHECK: call {{.*}}void @__kmpc_push_proc_bind([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID_VAL]], i32 2)
// CHECK: call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: ret void

// CHECK: !{!"llvm.loop.vectorize.enable", i1 true}

#endif
