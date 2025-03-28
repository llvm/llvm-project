// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void func1() {
#pragma omp metadirective when(user = {condition(0)}	\
			       : parallel for) otherwise()
  for (int i = 0; i < 100; i++)
    ;

#pragma omp metadirective when(user = {condition(0)}	\
			       : parallel for)
  for (int i = 0; i < 100; i++)
    ;

#pragma omp metadirective when(user = {condition(0)}			\
			       : parallel for)                          \
  when(implementation = {extension(match_none)}				\
       : parallel) default(parallel for)

  for (int i = 0; i < 100; i++)
    ;


}

// CHECK-LABEL: define dso_local void @_Z5func1v()
// CHECK:       entry
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I1:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i32 0, ptr [[I]], align 4
// CHECK-NEXT:    br label %[[FOR_COND:.*]]
// CHECK:       [[FOR_COND]]:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[I]], align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 100
// CHECK-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END:.*]]
// CHECK:       [[FOR_BODY]]:
// CHECK-NEXT:    br label %[[FOR_INC:.*]]
// CHECK:       [[FOR_INC]]:
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[I]], align 4
// CHECK-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP1]], 1
// CHECK-NEXT:    store i32 [[INC]], ptr [[I]], align 4
// CHECK-NEXT:    br label %[[FOR_COND]], !llvm.loop [[LOOP3:![0-9]+]]
// CHECK:       [[FOR_END]]:
// CHECK-NEXT:    store i32 0, ptr [[I1]], align 4
// CHECK-NEXT:    br label %[[FOR_COND2:.*]]
// CHECK:       [[FOR_COND2]]:
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[I1]], align 4
// CHECK-NEXT:    [[CMP3:%.*]] = icmp slt i32 [[TMP2]], 100
// CHECK-NEXT:    br i1 [[CMP3]], label %[[FOR_BODY4:.*]], label %[[FOR_END7:.*]]
// CHECK:       [[FOR_BODY4]]:
// CHECK-NEXT:    br label %[[FOR_INC5:.*]]
// CHECK:       [[FOR_INC5]]:
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[I1]], align 4
// CHECK-NEXT:    [[INC6:%.*]] = add nsw i32 [[TMP3]], 1
// CHECK-NEXT:    store i32 [[INC6]], ptr [[I1]], align 4
// CHECK-NEXT:    br label %[[FOR_COND2]], !llvm.loop [[LOOP5:![0-9]+]]
// CHECK:       [[FOR_END7]]:
// CHECK:    ret void

void func2() {
#pragma omp metadirective when(user = {condition(1)}	\
			       : parallel for) otherwise()
  for (int i = 0; i < 100; i++)
    ;

#pragma omp metadirective when(user = {condition(1)}	\
			       : parallel for)
  for (int i = 0; i < 100; i++)
    ;
}

// CHECK-LABEL: define dso_local void @_Z5func2v()
// CHECK:       entry
// CHECK-NEXT:    call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @[[GLOB2:[0-9]+]], i32 0, ptr @_Z5func2v.omp_outlined)
// CHECK-NEXT:    call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @[[GLOB2]], i32 0, ptr @_Z5func2v.omp_outlined.1)
// CHECK-NEXT:    ret void


void func3() {
#pragma omp metadirective when(user = {condition(0)}			\
			       : parallel for)                          \
  when(implementation = {extension(match_none)}				\
       : parallel) default(parallel for)

  for (int i = 0; i < 100; i++)
    ;

}

// CHECK-LABEL: define dso_local void @_Z5func3v()
// CHECK:       entry
// CHECK-NEXT:    call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 0, ptr @_Z5func3v.omp_outlined)
// CHECK-NEXT:    ret void
// CHECK-NEXT:   }

// CHECK-LABEL: define internal void @_Z5func3v.omp_outlined
// CHECK-SAME: (ptr noalias noundef [[DOTGLOBAL_TID_:%.*]],
// CHECK-SAME:  ptr noalias noundef [[DOTBOUND_TID_:%.*]])
// CHECK-NEXT:  entry
// CHECK-NEXT:    [[GLOB_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[BOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[GLOB_TID__ADDR]], align 8
// CHECK-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[BOUND_TID__ADDR]], align 8
// CHECK-NEXT:    store i32 0, ptr [[I]], align 4
// CHECK-NEXT:    br label %for.cond
// CHECK:for.cond:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[I]], align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 100
// CHECK-NEXT:    br i1 [[CMP]], label [[FOR_BODY:%.*]], label [[FOR_END:%.*]]
// CHECK:for.body:
// CHECK-NEXT:    br label [[FOR_INC:%.*]]
// CHECK:for.inc:
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[I]], align 4
// CHECK-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP1]], 1
// CHECK-NEXT:    store i32 [[INC]], ptr [[I]], align 4
// CHECK-NEXT:    br label [[FOR_COND:%.*]]
// CHECK:for.end:
// CHECK-NEXT:  ret void
// CHECK-NEXT:}

#endif
