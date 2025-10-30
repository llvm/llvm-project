// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s

// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=60 -DOMP60 -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK6
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -DOMP60 -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -DOMP60 -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK6
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-LABEL: @main
int main(int argc, char **argv) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[DEFLOC:@.+]])
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr [[DEFLOC]], i32 [[GTID]],
// CHECK: call i32 @__kmpc_omp_task(ptr [[DEFLOC]], i32 [[GTID]],
#pragma omp task
  ;
// CHECK: call void @__kmpc_taskgroup(ptr [[DEFLOC]], i32 [[GTID]])
// CHECK: [[TASKV:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr [[DEFLOC]], i32 [[GTID]], i32 33, i64 80, i64 1, ptr [[TASK1:@.+]])
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds nuw %{{.+}}, ptr [[TASKV]], i32 0, i32 0
// CHECK: [[DOWN:%.+]] = getelementptr inbounds nuw [[TD_TY:%.+]], ptr [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, ptr [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 9, ptr [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, ptr [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, ptr [[ST]],
// CHECK: call void @__kmpc_taskloop(ptr [[DEFLOC]], i32 [[GTID]], ptr [[TASKV]], i32 1, ptr [[DOWN]], ptr [[UP]], i64 [[ST_VAL]], i32 1, i32 0, i64 0, ptr null)
// CHECK: call void @__kmpc_end_taskgroup(ptr [[DEFLOC]], i32 [[GTID]])
#pragma omp taskloop priority(argc)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: [[TASKV:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 1, ptr [[TASK2:@.+]])
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds nuw %{{.+}}, ptr [[TASKV]], i32 0, i32 0
// CHECK: [[DOWN:%.+]] = getelementptr inbounds nuw [[TD_TY:%.+]], ptr [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, ptr [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 9, ptr [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, ptr [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, ptr [[ST]],
// CHECK: [[GRAINSIZE:%.+]] = zext i32 %{{.+}} to i64
// CHECK: call void @__kmpc_taskloop(ptr [[DEFLOC]], i32 [[GTID]], ptr [[TASKV]], i32 1, ptr [[DOWN]], ptr [[UP]], i64 [[ST_VAL]], i32 1, i32 1, i64 [[GRAINSIZE]], ptr null)
#pragma omp taskloop nogroup grainsize(argc)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: call void @__kmpc_taskgroup(ptr [[DEFLOC]], i32 [[GTID]])
// CHECK: [[TASKV:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 16, ptr [[TASK3:@.+]])
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds nuw %{{.+}}, ptr [[TASKV]], i32 0, i32 0
// CHECK: [[IF:%.+]] = icmp ne i32 %{{.+}}, 0
// CHECK: [[IF_INT:%.+]] = sext i1 [[IF]] to i32
// CHECK: [[DOWN:%.+]] = getelementptr inbounds nuw [[TD_TY:%.+]], ptr [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, ptr [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 %{{.+}}, ptr [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, ptr [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, ptr [[ST]],
// CHECK: call void @__kmpc_taskloop(ptr [[DEFLOC]], i32 [[GTID]], ptr [[TASKV]], i32 [[IF_INT]], ptr [[DOWN]], ptr [[UP]], i64 [[ST_VAL]], i32 1, i32 2, i64 4, ptr null)
// CHECK: call void @__kmpc_end_taskgroup(ptr [[DEFLOC]], i32 [[GTID]])
  int i;
#pragma omp taskloop if(argc) shared(argc, argv) collapse(2) num_tasks(4)
  for (i = 0; i < argc; ++i)
  for (int j = argc; j < argv[argc][argc]; ++j)
    ;
// CHECK: call void @__kmpc_taskgroup(
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %{{.+}}, i32 1, i64 80, i64 1, ptr [[TASK_CANCEL:@.+]])
// CHECK: call void @__kmpc_taskloop(
// CHECK: call void @__kmpc_end_taskgroup(
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp cancel taskgroup
#pragma omp cancellation point taskgroup
  }
}

// CHECK: define internal noundef i32 [[TASK1]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds nuw [[TD_TY:%.+]], ptr %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, ptr [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, ptr [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, ptr [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, ptr [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], ptr [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], ptr [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], ptr [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], ptr [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, ptr [[LB]],
// CHECK: [[LB_I32:%.+]] = trunc i64 [[LB_VAL]] to i32
// CHECK: store i32 [[LB_I32]], ptr [[CNT:%.+]],
// CHECK: br label
// CHECK: [[VAL:%.+]] = load i32, ptr [[CNT]],
// CHECK: [[VAL_I64:%.+]] = sext i32 [[VAL]] to i64
// CHECK: [[UB_VAL:%.+]] = load i64, ptr [[UB]],
// CHECK: [[CMP:%.+]] = icmp ule i64 [[VAL_I64]], [[UB_VAL]]
// CHECK: br i1 [[CMP]], label %{{.+}}, label %{{.+}}
// CHECK: load i32, ptr %
// CHECK: store i32 %
// CHECK: load i32, ptr %
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i32 %{{.+}}, ptr %
// CHECK: br label %
// CHECK: ret i32 0

// CHECK: define internal noundef i32 [[TASK2]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds nuw [[TD_TY:%.+]], ptr %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, ptr [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, ptr [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, ptr [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, ptr [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], ptr [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], ptr [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], ptr [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], ptr [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, ptr [[LB]],
// CHECK: [[LB_I32:%.+]] = trunc i64 [[LB_VAL]] to i32
// CHECK: store i32 [[LB_I32]], ptr [[CNT:%.+]],
// CHECK: br label
// CHECK: [[VAL:%.+]] = load i32, ptr [[CNT]],
// CHECK: [[VAL_I64:%.+]] = sext i32 [[VAL]] to i64
// CHECK: [[UB_VAL:%.+]] = load i64, ptr [[UB]],
// CHECK: [[CMP:%.+]] = icmp ule i64 [[VAL_I64]], [[UB_VAL]]
// CHECK: br i1 [[CMP]], label %{{.+}}, label %{{.+}}
// CHECK: load i32, ptr %
// CHECK: store i32 %
// CHECK: load i32, ptr %
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i32 %{{.+}}, ptr %
// CHECK: br label %
// CHECK: ret i32 0

// CHECK: define internal noundef i32 [[TASK3]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds nuw [[TD_TY:%.+]], ptr %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, ptr [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, ptr [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, ptr [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, ptr [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], ptr [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], ptr [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], ptr [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], ptr [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, ptr [[LB]],
// CHECK: store i64 [[LB_VAL]], ptr [[CNT:%.+]],
// CHECK: br label
// CHECK: ret i32 0

// CHECK: define internal noundef i32 [[TASK_CANCEL]](
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel(ptr @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK: [[IS_CANCEL:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[IS_CANCEL]], label %[[EXIT:.+]], label %[[CONTINUE:[^,]+]]
// CHECK: [[EXIT]]:
// CHECK: store i32 1, ptr [[CLEANUP_SLOT:%.+]],
// CHECK: br label %[[DONE:[^,]+]]
// CHECK: [[CONTINUE]]:
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(ptr @{{.+}}, i32 %{{.+}}, i32 4)
// CHECK: [[IS_CANCEL:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[IS_CANCEL]], label %[[EXIT2:.+]], label %[[CONTINUE2:[^,]+]]
// CHECK: [[EXIT2]]:
// CHECK: store i32 1, ptr [[CLEANUP_SLOT]],
// CHECK: br label %[[DONE]]
// CHECK: store i32 0, ptr [[CLEANUP_SLOT]],
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]:
// CHECK: ret i32 0

// CHECK-LABEL: @_ZN1SC2Ei
struct S {
  int a;
  S(int c) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[DEFLOC:@.+]])
// CHECK: [[TASKV:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr [[DEFLOC]], i32 [[GTID]], i32 1, i64 80, i64 16, ptr [[TASK4:@.+]])
// CHECK: [[TASK_DATA:%.+]] = getelementptr inbounds nuw %{{.+}}, ptr [[TASKV]], i32 0, i32 0
// CHECK: [[DOWN:%.+]] = getelementptr inbounds nuw [[TD_TY:%.+]], ptr [[TASK_DATA]], i32 0, i32 5
// CHECK: store i64 0, ptr [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr [[TASK_DATA]], i32 0, i32 6
// CHECK: store i64 %{{.+}}, ptr [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr [[TASK_DATA]], i32 0, i32 7
// CHECK: store i64 1, ptr [[ST]],
// CHECK: [[ST_VAL:%.+]] = load i64, ptr [[ST]],
// CHECK: [[NUM_TASKS:%.+]] = zext i32 %{{.+}} to i64
// CHECK: call void @__kmpc_taskloop(ptr [[DEFLOC]], i32 [[GTID]], ptr [[TASKV]], i32 1, ptr [[DOWN]], ptr [[UP]], i64 [[ST_VAL]], i32 1, i32 2, i64 [[NUM_TASKS]], ptr null)
#pragma omp taskloop shared(c) num_tasks(a)
    for (a = 0; a < c; ++a)
      ;
  }
} s(1);

// CHECK: define internal noundef i32 [[TASK4]](
// CHECK: [[DOWN:%.+]] = getelementptr inbounds nuw [[TD_TY:%.+]], ptr %{{.+}}, i32 0, i32 5
// CHECK: [[DOWN_VAL:%.+]] = load i64, ptr [[DOWN]],
// CHECK: [[UP:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 6
// CHECK: [[UP_VAL:%.+]] = load i64, ptr [[UP]],
// CHECK: [[ST:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 7
// CHECK: [[ST_VAL:%.+]] = load i64, ptr [[ST]],
// CHECK: [[LITER:%.+]] = getelementptr inbounds nuw [[TD_TY]], ptr %{{.+}}, i32 0, i32 8
// CHECK: [[LITER_VAL:%.+]] = load i32, ptr [[LITER]],
// CHECK: store i64 [[DOWN_VAL]], ptr [[LB:%[^,]+]],
// CHECK: store i64 [[UP_VAL]], ptr [[UB:%[^,]+]],
// CHECK: store i64 [[ST_VAL]], ptr [[ST:%[^,]+]],
// CHECK: store i32 [[LITER_VAL]], ptr [[LITER:%[^,]+]],
// CHECK: [[LB_VAL:%.+]] = load i64, ptr [[LB]],
// CHECK: [[LB_I32:%.+]] = trunc i64 [[LB_VAL]] to i32
// CHECK: store i32 [[LB_I32]], ptr [[CNT:%.+]],
// CHECK: br label
// CHECK: [[VAL:%.+]] = load i32, ptr [[CNT]],
// CHECK: [[VAL_I64:%.+]] = sext i32 [[VAL]] to i64
// CHECK: [[UB_VAL:%.+]] = load i64, ptr [[UB]],
// CHECK: [[CMP:%.+]] = icmp ule i64 [[VAL_I64]], [[UB_VAL]]
// CHECK: br i1 [[CMP]], label %{{.+}}, label %{{.+}}
// CHECK: load i32, ptr %
// CHECK: store i32 %
// CHECK: load i32, ptr %
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: store i32 %{{.+}}, ptr %
// CHECK: br label %
// CHECK: ret i32 0

class St {
public:
  operator int();
  St &operator+=(int);
};

// CHECK-LABEL: taskloop_with_class
void taskloop_with_class() {
  St s1;
  // CHECK: [[TD:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 [[GTID:%.+]], i32 1, i64 88, i64 8, ptr @{{.+}})
  // CHECK: call void @__kmpc_taskloop(ptr @{{.+}}, i32 [[GTID]], ptr [[TD]], i32 1, ptr %{{.+}}, ptr %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, ptr @{{.+}})
#pragma omp taskloop
  for (St s = St(); s < s1; s += 1) {
  }
}

#ifdef OMP60
void test_threadset()
{
#pragma omp taskloop threadset(omp_team)
  for (int i = 0; i < 10; ++i) {
  }
#pragma omp taskloop threadset(omp_pool)
  for (int i = 0; i < 10; ++i) {
  }
}
#endif // OMP60
// CHECK6-LABEL: define void @_Z14test_threadsetv()
// CHECK6-NEXT:  entry:
// CHECK6-NEXT:       [[AGG_CAPTURED:%.*]] = alloca [[STRUCT_ANON_14:%.*]], align 1
// CHECK6-NEXT:       %[[TMP:.*]] = alloca i32, align 4
// CHECK6-NEXT:       [[AGG_CAPTURED1:%.*]] = alloca [[STRUCT_ANON_16:%.*]], align 1
// CHECK6-NEXT:       %[[TMP2:.*]] = alloca i32, align 4
// CHECK6-NEXT:       %[[TID0:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB_PTR:[0-9]+]])
// CHECK6-NEXT:       call void @__kmpc_taskgroup(ptr @1, i32 %[[TID0:.*]])
// CHECK6-NEXT:       %[[TID1:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[TID0:.*]], i32 1, i64 80, i64 1, ptr @.omp_task_entry..[[ENTRY1:[0-9]+]])
// CHECK6-NEXT:       %[[TID2:.*]] = getelementptr inbounds nuw %struct.kmp_task_t_with_privates{{.*}}, ptr %[[TID1:.*]], i32 0, i32 0
// CHECK6-NEXT:       %[[TID3:.*]] = getelementptr inbounds nuw %struct.kmp_task_t{{.*}}, ptr %[[TID2:.*]], i32 0, i32 5
// CHECK6-NEXT:       store i64 0, ptr %[[TID3:.*]], align 8
// CHECK6-NEXT:       %[[TID4:.*]] = getelementptr inbounds nuw %struct.kmp_task_t{{.*}}, ptr %[[TID2:.*]], i32 0, i32 6
// CHECK6-NEXT:       store i64 9, ptr %[[TID4:.*]], align 8
// CHECK6-NEXT:       %[[TID5:.*]] = getelementptr inbounds nuw %struct.kmp_task_t{{.*}}, ptr %[[TID2:.*]], i32 0, i32 7
// CHECK6-NEXT:       store i64 1, ptr %[[TID5:.*]], align 8
// CHECK6-NEXT:       %[[TID6:.*]] = getelementptr inbounds nuw %struct.kmp_task_t{{.*}}, ptr %[[TID2:.*]], i32 0, i32 9
// CHECK6-NEXT:       call void @llvm.memset.p0.i64(ptr align 8 %[[TID6:.*]], i8 0, i64 8, i1 false)
// CHECK6-NEXT:       %[[TID7:.*]] = load i64, ptr %[[TID5:.*]], align 8
// CHECK6-NEXT:       call void @__kmpc_taskloop(ptr @1, i32 %[[TID0:.*]], ptr %[[TID1:.*]], i32 1, ptr %[[TID3:.*]], ptr %4, i64 %[[TID7:.*]], i32 1, i32 0, i64 0, ptr null)
// CHECK6-NEXT:       call void @__kmpc_end_taskgroup(ptr @1, i32 %[[TID0:.*]])
// CHECK6-NEXT:       call void @__kmpc_taskgroup(ptr @1, i32 %[[TID0:.*]])
// CHECK6-NEXT:       %[[TID8:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[TID0:.*]], i32 129, i64 80, i64 1, ptr @.omp_task_entry..[[ENTRY1:[0-9]+]])
// CHECK6-NEXT:       %[[TID9:.*]] = getelementptr inbounds nuw %struct.kmp_task_t_with_privates{{.*}}, ptr %[[TID8:.*]], i32 0, i32 0
// CHECK6-NEXT:       %[[TID10:.*]] = getelementptr inbounds nuw %struct.kmp_task_t{{.*}}, ptr %[[TID9:.*]], i32 0, i32 5
// CHECK6-NEXT:       store i64 0, ptr %[[TID10:.*]], align 8
// CHECK6-NEXT:       %[[TID11:.*]] = getelementptr inbounds nuw %struct.kmp_task_t{{.*}}, ptr %[[TID9:.*]], i32 0, i32 6
// CHECK6-NEXT:       store i64 9, ptr %[[TID11:.*]], align 8
// CHECK6-NEXT:       %[[TID12:.*]] = getelementptr inbounds nuw %struct.kmp_task_t{{.*}}, ptr %[[TID9:.*]], i32 0, i32 7
// CHECK6-NEXT:       store i64 1, ptr %[[TID12:.*]], align 8
// CHECK6-NEXT:       %[[TID13:.*]] = getelementptr inbounds nuw %struct.kmp_task_t{{.*}}, ptr %[[TID9:.*]], i32 0, i32 9
// CHECK6-NEXT:       call void @llvm.memset.p0.i64(ptr align 8 [[TID13:.*]], i8 0, i64 8, i1 false)
// CHECK6-NEXT:       %[[TID14:.*]] = load i64, ptr [[TID12:.*]], align 8
// CHECK6-NEXT:       call void @__kmpc_taskloop(ptr @1, i32 %[[TID0:.*]], ptr %[[TID8:.*]], i32 1, ptr %[[TID10:.*]], ptr %[[TID11:.*]], i64 %[[TID14:.*]], i32 1, i32 0, i64 0, ptr null)
// CHECK6-NEXT:       call void @__kmpc_end_taskgroup(ptr @1, i32 %[[TID0:.*]])
// CHECK6-NEXT:       ret void

#endif
