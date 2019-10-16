// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - -femit-all-decls | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -x c++ -emit-llvm %s -o - -femit-all-decls | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-LABEL: @main
int main(int argc, char **argv) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[DEFLOC:@.+]])
// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEFLOC]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64)* [[OMP_OUTLINED1:@.+]] to void (i32*, i32*, ...)*), i64 [[PRIORITY:%.+]])
// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEFLOC]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64)* [[OMP_OUTLINED2:@.+]] to void (i32*, i32*, ...)*), i64 [[GRAINSIZE:%.+]])
// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEFLOC]], i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*, i8***, i64, i64)* [[OMP_OUTLINED3:@.+]] to void (i32*, i32*, ...)*), i32* [[ARGC:%.+]], i8*** [[ARGV:%.+]], i64 [[COND:%.+]], i64 [[NUM_TASKS:%.+]])
// CHECK: call void @__kmpc_serialized_parallel(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK: call void [[OMP_OUTLINED3]](i32* %{{.+}}, i32* %{{.+}}, i32* [[ARGC]], i8*** [[ARGV]], i64 [[COND]], i64 [[NUM_TASKS]])
// CHECK: call void @__kmpc_end_serialized_parallel(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])


// CHECK: define internal void [[OMP_OUTLINED1]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i64 %{{.+}})
// CHECK: [[PRIO_ADDR:%.+]] = bitcast i64* %{{.+}} to i32*
// CHECK:       [[RES:%.+]] = call {{.*}}i32 @__kmpc_master(%struct.ident_t* [[DEFLOC]], i32 [[GTID:%.+]])
// CHECK-NEXT:  [[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       [[THEN]]
// CHECK: call void @__kmpc_taskgroup(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK: [[PRIO:%.+]] = load i32, i32* [[PRIO_ADDR]],
// CHECK: [[TASKV:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i32 33, i64 80, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[TDP_TY:%.+]]*)* [[TASK1:@.+]] to i32 (i32, i8*)*))
// CHECK: [[TASK:%.+]] = bitcast i8* [[TASKV]] to [[TDP_TY]]*
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds [[TDP_TY]], [[TDP_TY]]* [[TASK]], i32 0, i32 0
// CHECK: [[PRIO_ADDR:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 4
// CHECK: [[PRIO_ADDR_CAST:%.+]] = bitcast %{{.+}}* [[PRIO_ADDR]] to i32*
// CHECK: store i32 [[PRIO]], i32* [[PRIO_ADDR_CAST]],
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 9, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, i64* [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i8* [[TASKV]], i32 1, i64* [[DOWN]], i64* [[UP]], i64 [[ST_VAL]], i32 1, i32 0, i64 0, i8* null)
// CHECK: call void @__kmpc_end_taskgroup(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK-NEXT:  call {{.*}}void @__kmpc_end_master(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]


// CHECK: define internal i32 [[TASK1]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, i32* [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], i64* [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], i64* [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], i64* [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], i32* [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: [[LB_I32:%.+]] = trunc i64 [[LB_VAL]] to i32
// CHECK: store i32 [[LB_I32]], i32* [[CNT:%.+]],
// CHECK: br label
// CHECK: [[VAL:%.+]] = load i32, i32* [[CNT]],
// CHECK: [[VAL_I64:%.+]] = sext i32 [[VAL]] to i64
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp ule i64 [[VAL_I64]], [[UB_VAL]]
// CHECK: br i1 [[CMP]], label %{{.+}}, label %{{.+}}
// CHECK: load i32, i32* %
// CHECK: store i32 %
// CHECK: load i32, i32* %
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i32 %{{.+}}, i32* %
// CHECK: br label %
// CHECK: ret i32 0

#pragma omp parallel master taskloop priority(argc)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: define internal void [[OMP_OUTLINED2]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i64 %{{.+}})
// CHECK:       [[RES:%.+]] = call {{.*}}i32 @__kmpc_master(%struct.ident_t* [[DEFLOC]], i32 [[GTID:%.+]])
// CHECK-NEXT:  [[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       [[THEN]]
// CHECK: [[TASKV:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[TDP_TY:%.+]]*)* [[TASK2:@.+]] to i32 (i32, i8*)*))
// CHECK: [[TASK:%.+]] = bitcast i8* [[TASKV]] to [[TDP_TY]]*
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds [[TDP_TY]], [[TDP_TY]]* [[TASK]], i32 0, i32 0
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 9, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, i64* [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[GRAINSIZE:%.+]] = zext i32 %{{.+}} to i64
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i8* [[TASKV]], i32 1, i64* [[DOWN]], i64* [[UP]], i64 [[ST_VAL]], i32 1, i32 1, i64 [[GRAINSIZE]], i8* null)
// CHECK-NEXT:  call {{.*}}void @__kmpc_end_master(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]


// CHECK: define internal i32 [[TASK2]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, i32* [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], i64* [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], i64* [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], i64* [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], i32* [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: [[LB_I32:%.+]] = trunc i64 [[LB_VAL]] to i32
// CHECK: store i32 [[LB_I32]], i32* [[CNT:%.+]],
// CHECK: br label
// CHECK: [[VAL:%.+]] = load i32, i32* [[CNT]],
// CHECK: [[VAL_I64:%.+]] = sext i32 [[VAL]] to i64
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp ule i64 [[VAL_I64]], [[UB_VAL]]
// CHECK: br i1 [[CMP]], label %{{.+}}, label %{{.+}}
// CHECK: load i32, i32* %
// CHECK: store i32 %
// CHECK: load i32, i32* %
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i32 %{{.+}}, i32* %
// CHECK: br label %
// CHECK: ret i32 0

#pragma omp parallel master taskloop nogroup grainsize(argc)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: define internal void [[OMP_OUTLINED3]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}}, i8*** dereferenceable(8) %{{.+}}, i64 %{{.+}}, i64 %{{.+}})
// CHECK:       [[RES:%.+]] = call {{.*}}i32 @__kmpc_master(%struct.ident_t* [[DEFLOC]], i32 [[GTID:%.+]])
// CHECK-NEXT:  [[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       [[THEN]]
// CHECK: call void @__kmpc_taskgroup(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK: [[TASKV:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 16, i32 (i32, i8*)* bitcast (i32 (i32, [[TDP_TY:%.+]]*)* [[TASK3:@.+]] to i32 (i32, i8*)*))
// CHECK: [[TASK:%.+]] = bitcast i8* [[TASKV]] to [[TDP_TY]]*
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds [[TDP_TY]], [[TDP_TY]]* [[TASK]], i32 0, i32 0
// CHECK: [[COND_VAL:%.+]] = load i8, i8* %{{.+}},
// CHECK: [[COND_BOOL:%.+]] = trunc i8 [[COND_VAL]] to i1
// CHECK: [[IF_INT:%.+]] = sext i1 [[COND_BOOL]] to i32
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 %{{.+}}, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, i64* [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[NUM_TASKS:%.+]] = zext i32 %{{.+}} to i64
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i8* [[TASKV]], i32 [[IF_INT]], i64* [[DOWN]], i64* [[UP]], i64 [[ST_VAL]], i32 1, i32 2, i64 [[NUM_TASKS]], i8* null)
// CHECK: call void @__kmpc_end_taskgroup(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK-NEXT:  call {{.*}}void @__kmpc_end_master(%struct.ident_t* [[DEFLOC]], i32 [[GTID]])
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]

// CHECK: define internal i32 [[TASK3]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, i32* [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], i64* [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], i64* [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], i64* [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], i32* [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: store i64 [[LB_VAL]], i64* [[CNT:%.+]],
// CHECK: br label
// CHECK: ret i32 0

  int i;
#pragma omp parallel master taskloop if(argc) shared(argc, argv) collapse(2) num_tasks(argc)
  for (i = 0; i < argc; ++i)
  for (int j = argc; j < argv[argc][argc]; ++j)
    ;
}

// CHECK-LABEL: @_ZN1SC2Ei
struct S {
  int a;
  S(int c) {
// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEFLOC]], i32 3, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, %struct.S*, i32*, i64)* [[OMP_OUTLINED4:@.+]] to void (i32*, i32*, ...)*), %struct.S* %{{.+}}, i32* %{{.+}}, i64 %{{.+}})

// CHECK: define internal void [[OMP_OUTLINED4]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, %struct.S* %{{.+}}, i32* dereferenceable(4) %{{.+}}, i64 %{{.+}})
// CHECK: [[CONV:%.+]] = bitcast i64* %{{.+}} to i8*
// CHECK: [[CONDI8:%.+]] = load i8, i8* [[CONV]],
// CHECK: [[COND:%.+]] = trunc i8 [[CONDI8]] to i1
// CHECK: [[IS_FINAL:%.+]] = select i1 [[COND:%.+]], i32 2, i32 0
// CHECK: [[FLAGS:%.+]] = or i32 [[IS_FINAL]], 1
// CHECK: [[TASKV:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* [[DEFLOC]], i32 [[GTID:%.+]], i32 [[FLAGS]], i64 80, i64 16, i32 (i32, i8*)* bitcast (i32 (i32, [[TDP_TY:%.+]]*)* [[TASK4:@.+]] to i32 (i32, i8*)*))
// CHECK: [[TASK:%.+]] = bitcast i8* [[TASKV]] to [[TDP_TY]]*
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds [[TDP_TY]], [[TDP_TY]]* [[TASK]], i32 0, i32 0
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 %{{.+}}, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, i64* [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* [[DEFLOC]], i32 [[GTID]], i8* [[TASKV]], i32 1, i64* [[DOWN]], i64* [[UP]], i64 [[ST_VAL]], i32 1, i32 2, i64 4, i8* null)
#pragma omp parallel master taskloop shared(c) num_tasks(4) final(c)
    for (a = 0; a < c; ++a)
      ;
  }
} s(1);

// CHECK: define internal i32 [[TASK4]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds [[TD_TY:%.+]], [[TD_TY]]* %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, i64* [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, i64* [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, i64* [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds [[TD_TY]], [[TD_TY]]* %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, i32* [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], i64* [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], i64* [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], i64* [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], i32* [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: [[LB_I32:%.+]] = trunc i64 [[LB_VAL]] to i32
// CHECK: store i32 [[LB_I32]], i32* [[CNT:%.+]],
// CHECK: br label
// CHECK: [[VAL:%.+]] = load i32, i32* [[CNT]],
// CHECK: [[VAL_I64:%.+]] = sext i32 [[VAL]] to i64
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp ule i64 [[VAL_I64]], [[UB_VAL]]
// CHECK: br i1 [[CMP]], label %{{.+}}, label %{{.+}}
// CHECK: load i32, i32* %
// CHECK: store i32 %
// CHECK: load i32, i32* %
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i32 %{{.+}}, i32* %
// CHECK: br label %
// CHECK: ret i32 0

#endif
