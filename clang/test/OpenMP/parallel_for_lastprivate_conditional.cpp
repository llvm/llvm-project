// RUN: %clang_cc1 -verify -fopenmp -DOMP5 -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -DOMP5 -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int main() {
  int a = 0;
#pragma omp parallel for lastprivate(conditional: a)
  for (int i = 0; i < 10; ++i) {
    if (i < 5) {
      a = 0;
#pragma omp parallel reduction(+:a) num_threads(10)
      a += i;
#pragma omp atomic
      a += i;
#pragma omp parallel num_threads(10)
#pragma omp atomic
      a += i;
    }
  }
  return 0;
}

// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @{{.+}}, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*)* [[OUTLINED:@.+]] to void (i32*, i32*, ...)*), i32* %{{.+}})

// CHECK: define internal void [[OUTLINED]](
// CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 10)
// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @{{.+}}, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*, i32*)* @{{.+}} to void (i32*, i32*, ...)*), i32* {{.+}} i32* %{{.+}})
// CHECK: call void @__kmpc_critical(%struct.ident_t* @{{.+}}, i32 %{{.+}}, [8 x i32]* @{{.+}})
// CHECK: [[LAST_IV_VAL:%.+]] = load i32, i32* [[LAST_IV:@.+]],
// CHECK: [[RES:%.+]] = icmp sle i32 [[LAST_IV_VAL]], [[IV:%.+]]
// CHECK: br i1 [[RES]], label %[[THEN:.+]], label %[[DONE:.+]]
// CHECK: [[THEN]]:
// CHECK: store i32 [[IV]], i32* [[LAST_IV]],
// CHECK: [[A_VAL:%.+]] = load i32, i32* [[A_PRIV:%.+]],
// CHECK: store i32 [[A_VAL]], i32* [[A_GLOB:@.+]],
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]:
// CHECK: call void @__kmpc_end_critical(%struct.ident_t* @{{.+}}, i32 %{{.+}}, [8 x i32]* @{{.+}})
// CHECK: atomicrmw add i32*
// CHECK: call void @__kmpc_critical(%struct.ident_t* @{{.+}}, i32 %{{.+}}, [8 x i32]* @{{.+}})
// CHECK: [[LAST_IV_VAL:%.+]] = load i32, i32* [[LAST_IV:@.+]],
// CHECK: [[RES:%.+]] = icmp sle i32 [[LAST_IV_VAL]], [[IV:%.+]]
// CHECK: br i1 [[RES]], label %[[THEN:.+]], label %[[DONE:.+]]
// CHECK: [[THEN]]:
// CHECK: store i32 [[IV]], i32* [[LAST_IV]],
// CHECK: [[A_VAL:%.+]] = load i32, i32* [[A_PRIV:%.+]],
// CHECK: store i32 [[A_VAL]], i32* [[A_GLOB:@.+]],
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]:
// CHECK: call void @__kmpc_end_critical(%struct.ident_t* @{{.+}}, i32 %{{.+}}, [8 x i32]* @{{.+}})
// CHECK: call void @__kmpc_push_num_threads(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 10)
// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @{{.+}}, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*, i32*)* [[OUTLINED2:@.+]] to void (i32*, i32*, ...)*), i32* {{.+}} i32* %{{.+}})
// CHECK: [[FIRED:%.+]] = getelementptr inbounds %struct.{{.+}}, %struct.{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 1
// CHECK: [[FIRED_VAL:%.+]] = load i8, i8* [[FIRED]],
// CHECK: [[CMP:%.+]] = icmp ne i8 [[FIRED_VAL]], 0
// CHECK: br i1 [[CMP]], label %[[CHECK_THEN:.+]], label %[[CHECK_DONE:.+]]
// CHECK: [[CHECK_THEN]]:
// CHECK: call void @__kmpc_critical(%struct.ident_t* @{{.+}}, i32 %{{.+}}, [8 x i32]* @{{.+}})
// CHECK: [[LAST_IV_VAL:%.+]] = load i32, i32* [[LAST_IV:@.+]],
// CHECK: [[RES:%.+]] = icmp sle i32 [[LAST_IV_VAL]], [[IV:%.+]]
// CHECK: br i1 [[RES]], label %[[THEN:.+]], label %[[DONE:.+]]
// CHECK: [[THEN]]:
// CHECK: store i32 [[IV]], i32* [[LAST_IV]],
// CHECK: [[A_VAL:%.+]] = load i32, i32* [[A_PRIV:%.+]],
// CHECK: store i32 [[A_VAL]], i32* [[A_GLOB:@.+]],
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]:
// CHECK: call void @__kmpc_end_critical(%struct.ident_t* @{{.+}}, i32 %{{.+}}, [8 x i32]* @{{.+}})
// CHECK: br label %[[CHECK_DONE]]
// CHECK: [[CHECK_DONE]]:
// CHECK: call void @__kmpc_for_static_fini(%struct.ident_t* @{{.+}}, i32 %{{.+}})
// CHECK: [[IS_LAST:%.+]] = load i32, i32* %{{.+}},
// CHECK: [[RES:%.+]] = icmp ne i32 [[IS_LAST]], 0
// CHECK: call void @__kmpc_barrier(%struct.ident_t* @{{.+}}, i32 %{{.+}})
// CHECK: br i1 [[RES]], label %[[THEN:.+]], label %[[DONE:.+]]
// CHECK: [[THEN]]:
// CHECK: [[A_VAL:%.+]] = load i32, i32* [[A_GLOB]],
// CHECK: store i32 [[A_VAL]], i32* [[A_PRIV]],
// CHECK: [[A_VAL:%.+]] = load i32, i32* [[A_PRIV]],
// CHECK: store i32 [[A_VAL]], i32* %{{.+}},
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]:
// CHECK: ret void

// CHECK: define internal void [[OUTLINED2]](i32* {{.+}}, i32* {{.+}}, i32* {{.+}}, i32* {{.+}})
// CHECK: atomicrmw add i32* [[A_SHARED:%.+]], i32 %{{.+}} monotonic
// CHECK: [[BASE:%.+]] = bitcast i32* [[A_SHARED]] to [[STRUCT:%struct[.].+]]*
// CHECK: [[FIRED:%.+]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[BASE]], i{{.+}} 0, i{{.+}} 1
// CHECK: store atomic volatile i8 1, i8* [[FIRED]] unordered,
// CHECK: ret void

#endif // HEADER
