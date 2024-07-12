// RUN: %clang_cc1 -no-enable-noundef-analysis -verify -Wno-vla -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK1
// RUN: %clang_cc1 -no-enable-noundef-analysis -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK1
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

struct S {
  int a;
  S() : a(0) {}
  S(const S &) {}
  S &operator=(const S &) { return *this; }
  ~S() {}
  friend S operator+(const S &a, const S &b) { return a; }
};

int main(int argc, char **argv) {
  int a;
  float b;
  S c[5];
  short d[argc];
#pragma omp taskgroup task_reduction(+ \
                                     : a, b, argc)
  {
#pragma omp taskgroup task_reduction(- \
                                     : c, d)
#pragma omp parallel
#pragma omp target in_reduction(+ \
                                : a)
    for (int i = 0; i < 5; i++)
      a += d[a];
  }
  return 0;
}

#endif
// CHECK1-LABEL: define {{[^@]+}}@main
// CHECK1-SAME: (i32 [[ARGC:%.*]], ptr [[ARGV:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
// CHECK1-NEXT:    [[ARGC_ADDR:%.*]] = alloca i32, align 4
// CHECK1-NEXT:    [[ARGV_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[A:%.*]] = alloca i32, align 4
// CHECK1-NEXT:    [[B:%.*]] = alloca float, align 4
// CHECK1-NEXT:    [[C:%.*]] = alloca [5 x %struct.S], align 16
// CHECK1-NEXT:    [[SAVED_STACK:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[__VLA_EXPR0:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[DOTRD_INPUT_:%.*]] = alloca [3 x %struct.kmp_taskred_input_t], align 8
// CHECK1-NEXT:    [[DOTTASK_RED_:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTRD_INPUT_3:%.*]] = alloca [2 x %struct.kmp_taskred_input_t.0], align 8
// CHECK1-NEXT:    [[DOTTASK_RED_6:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB1:[0-9]+]])
// CHECK1-NEXT:    store i32 0, ptr [[RETVAL]], align 4
// CHECK1-NEXT:    store i32 [[ARGC]], ptr [[ARGC_ADDR]], align 4
// CHECK1-NEXT:    store ptr [[ARGV]], ptr [[ARGV_ADDR]], align 8
// CHECK1-NEXT:    [[ARRAY_BEGIN:%.*]] = getelementptr inbounds [5 x %struct.S], ptr [[C]], i32 0, i32 0
// CHECK1-NEXT:    [[ARRAYCTOR_END:%.*]] = getelementptr inbounds [[STRUCT_S:%.*]], ptr [[ARRAY_BEGIN]], i64 5
// CHECK1-NEXT:    br label [[ARRAYCTOR_LOOP:%.*]]
// CHECK1:       arrayctor.loop:
// CHECK1-NEXT:    [[ARRAYCTOR_CUR:%.*]] = phi ptr [ [[ARRAY_BEGIN]], [[ENTRY:%.*]] ], [ [[ARRAYCTOR_NEXT:%.*]], [[ARRAYCTOR_LOOP]] ]
// CHECK1-NEXT:    call void @_ZN1SC1Ev(ptr nonnull align 4 dereferenceable(4) [[ARRAYCTOR_CUR]])
// CHECK1-NEXT:    [[ARRAYCTOR_NEXT]] = getelementptr inbounds [[STRUCT_S]], ptr [[ARRAYCTOR_CUR]], i64 1
// CHECK1-NEXT:    [[ARRAYCTOR_DONE:%.*]] = icmp eq ptr [[ARRAYCTOR_NEXT]], [[ARRAYCTOR_END]]
// CHECK1-NEXT:    br i1 [[ARRAYCTOR_DONE]], label [[ARRAYCTOR_CONT:%.*]], label [[ARRAYCTOR_LOOP]]
// CHECK1:       arrayctor.cont:
// CHECK1-NEXT:    [[TMP1:%.*]] = load i32, ptr [[ARGC_ADDR]], align 4
// CHECK1-NEXT:    [[TMP2:%.*]] = zext i32 [[TMP1]] to i64
// CHECK1-NEXT:    [[TMP3:%.*]] = call ptr @llvm.stacksave.p0()
// CHECK1-NEXT:    store ptr [[TMP3]], ptr [[SAVED_STACK]], align 8
// CHECK1-NEXT:    [[VLA:%.*]] = alloca i16, i64 [[TMP2]], align 16
// CHECK1-NEXT:    store i64 [[TMP2]], ptr [[__VLA_EXPR0]], align 8
// CHECK1-NEXT:    call void @__kmpc_taskgroup(ptr @[[GLOB1]], i32 [[TMP0]])
// CHECK1-NEXT:    [[DOTRD_INPUT_GEP_:%.*]] = getelementptr inbounds [3 x %struct.kmp_taskred_input_t], ptr [[DOTRD_INPUT_]], i64 0, i64 0
// CHECK1-NEXT:    [[TMP4:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T:%.*]], ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 0
// CHECK1-NEXT:    store ptr [[A]], ptr [[TMP4]], align 8
// CHECK1-NEXT:    [[TMP6:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 1
// CHECK1-NEXT:    store ptr [[A]], ptr [[TMP6]], align 8
// CHECK1-NEXT:    [[TMP8:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 2
// CHECK1-NEXT:    store i64 4, ptr [[TMP8]], align 8
// CHECK1-NEXT:    [[TMP9:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 3
// CHECK1-NEXT:    store ptr @.red_init., ptr [[TMP9]], align 8
// CHECK1-NEXT:    [[TMP10:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 4
// CHECK1-NEXT:    store ptr null, ptr [[TMP10]], align 8
// CHECK1-NEXT:    [[TMP11:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 5
// CHECK1-NEXT:    store ptr @.red_comb., ptr [[TMP11]], align 8
// CHECK1-NEXT:    [[TMP12:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 6
// CHECK1-NEXT:    call void @llvm.memset.p0.i64(ptr align 8 [[TMP12]], i8 0, i64 4, i1 false)
// CHECK1-NEXT:    [[DOTRD_INPUT_GEP_1:%.*]] = getelementptr inbounds [3 x %struct.kmp_taskred_input_t], ptr [[DOTRD_INPUT_]], i64 0, i64 1
// CHECK1-NEXT:    [[TMP14:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_1]], i32 0, i32 0
// CHECK1-NEXT:    store ptr [[B]], ptr [[TMP14]], align 8
// CHECK1-NEXT:    [[TMP16:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_1]], i32 0, i32 1
// CHECK1-NEXT:    store ptr [[B]], ptr [[TMP16]], align 8
// CHECK1-NEXT:    [[TMP18:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_1]], i32 0, i32 2
// CHECK1-NEXT:    store i64 4, ptr [[TMP18]], align 8
// CHECK1-NEXT:    [[TMP19:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_1]], i32 0, i32 3
// CHECK1-NEXT:    store ptr @.red_init..1, ptr [[TMP19]], align 8
// CHECK1-NEXT:    [[TMP20:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_1]], i32 0, i32 4
// CHECK1-NEXT:    store ptr null, ptr [[TMP20]], align 8
// CHECK1-NEXT:    [[TMP21:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_1]], i32 0, i32 5
// CHECK1-NEXT:    store ptr @.red_comb..2, ptr [[TMP21]], align 8
// CHECK1-NEXT:    [[TMP22:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_1]], i32 0, i32 6
// CHECK1-NEXT:    call void @llvm.memset.p0.i64(ptr align 8 [[TMP22]], i8 0, i64 4, i1 false)
// CHECK1-NEXT:    [[DOTRD_INPUT_GEP_2:%.*]] = getelementptr inbounds [3 x %struct.kmp_taskred_input_t], ptr [[DOTRD_INPUT_]], i64 0, i64 2
// CHECK1-NEXT:    [[TMP24:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_2]], i32 0, i32 0
// CHECK1-NEXT:    store ptr [[ARGC_ADDR]], ptr [[TMP24]], align 8
// CHECK1-NEXT:    [[TMP26:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_2]], i32 0, i32 1
// CHECK1-NEXT:    store ptr [[ARGC_ADDR]], ptr [[TMP26]], align 8
// CHECK1-NEXT:    [[TMP28:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_2]], i32 0, i32 2
// CHECK1-NEXT:    store i64 4, ptr [[TMP28]], align 8
// CHECK1-NEXT:    [[TMP29:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_2]], i32 0, i32 3
// CHECK1-NEXT:    store ptr @.red_init..3, ptr [[TMP29]], align 8
// CHECK1-NEXT:    [[TMP30:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_2]], i32 0, i32 4
// CHECK1-NEXT:    store ptr null, ptr [[TMP30]], align 8
// CHECK1-NEXT:    [[TMP31:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_2]], i32 0, i32 5
// CHECK1-NEXT:    store ptr @.red_comb..4, ptr [[TMP31]], align 8
// CHECK1-NEXT:    [[TMP32:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T]], ptr [[DOTRD_INPUT_GEP_2]], i32 0, i32 6
// CHECK1-NEXT:    call void @llvm.memset.p0.i64(ptr align 8 [[TMP32]], i8 0, i64 4, i1 false)
// CHECK1-NEXT:    [[TMP35:%.*]] = call ptr @__kmpc_taskred_init(i32 [[TMP0]], i32 3, ptr [[DOTRD_INPUT_]])
// CHECK1-NEXT:    store ptr [[TMP35]], ptr [[DOTTASK_RED_]], align 8
// CHECK1-NEXT:    call void @__kmpc_taskgroup(ptr @[[GLOB1]], i32 [[TMP0]])
// CHECK1-NEXT:    [[DOTRD_INPUT_GEP_4:%.*]] = getelementptr inbounds [2 x %struct.kmp_taskred_input_t.0], ptr [[DOTRD_INPUT_3]], i64 0, i64 0
// CHECK1-NEXT:    [[TMP36:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0:%.*]], ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 0
// CHECK1-NEXT:    store ptr [[C]], ptr [[TMP36]], align 8
// CHECK1-NEXT:    [[TMP38:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 1
// CHECK1-NEXT:    store ptr [[C]], ptr [[TMP38]], align 8
// CHECK1-NEXT:    [[TMP40:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 2
// CHECK1-NEXT:    store i64 20, ptr [[TMP40]], align 8
// CHECK1-NEXT:    [[TMP41:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 3
// CHECK1-NEXT:    store ptr @.red_init..5, ptr [[TMP41]], align 8
// CHECK1-NEXT:    [[TMP42:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 4
// CHECK1-NEXT:    store ptr @.red_fini., ptr [[TMP42]], align 8
// CHECK1-NEXT:    [[TMP43:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 5
// CHECK1-NEXT:    store ptr @.red_comb..6, ptr [[TMP43]], align 8
// CHECK1-NEXT:    [[TMP44:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 6
// CHECK1-NEXT:    call void @llvm.memset.p0.i64(ptr align 8 [[TMP44]], i8 0, i64 4, i1 false)
// CHECK1-NEXT:    [[DOTRD_INPUT_GEP_5:%.*]] = getelementptr inbounds [2 x %struct.kmp_taskred_input_t.0], ptr [[DOTRD_INPUT_3]], i64 0, i64 1
// CHECK1-NEXT:    [[TMP46:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_5]], i32 0, i32 0
// CHECK1-NEXT:    store ptr [[VLA]], ptr [[TMP46]], align 8
// CHECK1-NEXT:    [[TMP48:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_5]], i32 0, i32 1
// CHECK1-NEXT:    store ptr [[VLA]], ptr [[TMP48]], align 8
// CHECK1-NEXT:    [[TMP50:%.*]] = mul nuw i64 [[TMP2]], 2
// CHECK1-NEXT:    [[TMP51:%.*]] = udiv exact i64 [[TMP50]], ptrtoint (ptr getelementptr (i16, ptr null, i32 1) to i64)
// CHECK1-NEXT:    [[TMP52:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_5]], i32 0, i32 2
// CHECK1-NEXT:    store i64 [[TMP50]], ptr [[TMP52]], align 8
// CHECK1-NEXT:    [[TMP53:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_5]], i32 0, i32 3
// CHECK1-NEXT:    store ptr @.red_init..7, ptr [[TMP53]], align 8
// CHECK1-NEXT:    [[TMP54:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_5]], i32 0, i32 4
// CHECK1-NEXT:    store ptr null, ptr [[TMP54]], align 8
// CHECK1-NEXT:    [[TMP55:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_5]], i32 0, i32 5
// CHECK1-NEXT:    store ptr @.red_comb..8, ptr [[TMP55]], align 8
// CHECK1-NEXT:    [[TMP56:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASKRED_INPUT_T_0]], ptr [[DOTRD_INPUT_GEP_5]], i32 0, i32 6
// CHECK1-NEXT:    store i32 1, ptr [[TMP56]], align 8
// CHECK1-NEXT:    [[TMP58:%.*]] = call ptr @__kmpc_taskred_init(i32 [[TMP0]], i32 2, ptr [[DOTRD_INPUT_3]])
// CHECK1-NEXT:    store ptr [[TMP58]], ptr [[DOTTASK_RED_6]], align 8
// CHECK1-NEXT:    call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @[[GLOB1]], i32 4, ptr @main.omp_outlined, ptr [[A]], i64 [[TMP2]], ptr [[VLA]], ptr [[DOTTASK_RED_]])
// CHECK1-NEXT:    call void @__kmpc_end_taskgroup(ptr @[[GLOB1]], i32 [[TMP0]])
// CHECK1-NEXT:    call void @__kmpc_end_taskgroup(ptr @[[GLOB1]], i32 [[TMP0]])
// CHECK1-NEXT:    store i32 0, ptr [[RETVAL]], align 4
// CHECK1-NEXT:    [[TMP59:%.*]] = load ptr, ptr [[SAVED_STACK]], align 8
// CHECK1-NEXT:    call void @llvm.stackrestore.p0(ptr [[TMP59]])
// CHECK1-NEXT:    [[ARRAY_BEGIN7:%.*]] = getelementptr inbounds [5 x %struct.S], ptr [[C]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP60:%.*]] = getelementptr inbounds [[STRUCT_S]], ptr [[ARRAY_BEGIN7]], i64 5
// CHECK1-NEXT:    br label [[ARRAYDESTROY_BODY:%.*]]
// CHECK1:       arraydestroy.body:
// CHECK1-NEXT:    [[ARRAYDESTROY_ELEMENTPAST:%.*]] = phi ptr [ [[TMP60]], [[ARRAYCTOR_CONT]] ], [ [[ARRAYDESTROY_ELEMENT:%.*]], [[ARRAYDESTROY_BODY]] ]
// CHECK1-NEXT:    [[ARRAYDESTROY_ELEMENT]] = getelementptr inbounds [[STRUCT_S]], ptr [[ARRAYDESTROY_ELEMENTPAST]], i64 -1
// CHECK1-NEXT:    call void @_ZN1SD1Ev(ptr nonnull align 4 dereferenceable(4) [[ARRAYDESTROY_ELEMENT]]) #[[ATTR3:[0-9]+]]
// CHECK1-NEXT:    [[ARRAYDESTROY_DONE:%.*]] = icmp eq ptr [[ARRAYDESTROY_ELEMENT]], [[ARRAY_BEGIN7]]
// CHECK1-NEXT:    br i1 [[ARRAYDESTROY_DONE]], label [[ARRAYDESTROY_DONE8:%.*]], label [[ARRAYDESTROY_BODY]]
// CHECK1:       arraydestroy.done8:
// CHECK1-NEXT:    [[TMP61:%.*]] = load i32, ptr [[RETVAL]], align 4
// CHECK1-NEXT:    ret i32 [[TMP61]]
//
//
// CHECK1-LABEL: define {{[^@]+}}@_ZN1SC1Ev
// CHECK1-SAME: (ptr nonnull align 4 dereferenceable(4) [[THIS:%.*]]) unnamed_addr #[[ATTR1:[0-9]+]] align 2 {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    call void @_ZN1SC2Ev(ptr nonnull align 4 dereferenceable(4) [[THIS1]])
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_init.
// CHECK1-SAME: (ptr noalias [[TMP0:%.*]], ptr noalias [[TMP1:%.*]]) #[[ATTR5:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store i32 0, ptr [[TMP3]], align 4
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_comb.
// CHECK1-SAME: (ptr [[TMP0:%.*]], ptr [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP6:%.*]] = load i32, ptr [[TMP3]], align 4
// CHECK1-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TMP5]], align 4
// CHECK1-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP6]], [[TMP7]]
// CHECK1-NEXT:    store i32 [[ADD]], ptr [[TMP3]], align 4
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_init..1
// CHECK1-SAME: (ptr noalias [[TMP0:%.*]], ptr noalias [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store float 0.000000e+00, ptr [[TMP3]], align 4
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_comb..2
// CHECK1-SAME: (ptr [[TMP0:%.*]], ptr [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP6:%.*]] = load float, ptr [[TMP3]], align 4
// CHECK1-NEXT:    [[TMP7:%.*]] = load float, ptr [[TMP5]], align 4
// CHECK1-NEXT:    [[ADD:%.*]] = fadd float [[TMP6]], [[TMP7]]
// CHECK1-NEXT:    store float [[ADD]], ptr [[TMP3]], align 4
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_init..3
// CHECK1-SAME: (ptr noalias [[TMP0:%.*]], ptr noalias [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store i32 0, ptr [[TMP3]], align 4
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_comb..4
// CHECK1-SAME: (ptr [[TMP0:%.*]], ptr [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP6:%.*]] = load i32, ptr [[TMP3]], align 4
// CHECK1-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TMP5]], align 4
// CHECK1-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP6]], [[TMP7]]
// CHECK1-NEXT:    store i32 [[ADD]], ptr [[TMP3]], align 4
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_init..5
// CHECK1-SAME: (ptr noalias [[TMP0:%.*]], ptr noalias [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[ARRAY_BEGIN:%.*]] = getelementptr inbounds [5 x %struct.S], ptr [[TMP3]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP4:%.*]] = getelementptr [[STRUCT_S:%.*]], ptr [[ARRAY_BEGIN]], i64 5
// CHECK1-NEXT:    [[OMP_ARRAYINIT_ISEMPTY:%.*]] = icmp eq ptr [[ARRAY_BEGIN]], [[TMP4]]
// CHECK1-NEXT:    br i1 [[OMP_ARRAYINIT_ISEMPTY]], label [[OMP_ARRAYINIT_DONE:%.*]], label [[OMP_ARRAYINIT_BODY:%.*]]
// CHECK1:       omp.arrayinit.body:
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DESTELEMENTPAST:%.*]] = phi ptr [ [[ARRAY_BEGIN]], [[ENTRY:%.*]] ], [ [[OMP_ARRAYCPY_DEST_ELEMENT:%.*]], [[OMP_ARRAYINIT_BODY]] ]
// CHECK1-NEXT:    call void @_ZN1SC1Ev(ptr nonnull align 4 dereferenceable(4) [[OMP_ARRAYCPY_DESTELEMENTPAST]])
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DEST_ELEMENT]] = getelementptr [[STRUCT_S]], ptr [[OMP_ARRAYCPY_DESTELEMENTPAST]], i32 1
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DONE:%.*]] = icmp eq ptr [[OMP_ARRAYCPY_DEST_ELEMENT]], [[TMP4]]
// CHECK1-NEXT:    br i1 [[OMP_ARRAYCPY_DONE]], label [[OMP_ARRAYINIT_DONE]], label [[OMP_ARRAYINIT_BODY]]
// CHECK1:       omp.arrayinit.done:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_fini.
// CHECK1-SAME: (ptr [[TMP0:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[ARRAY_BEGIN:%.*]] = getelementptr inbounds [5 x %struct.S], ptr [[TMP1]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP3:%.*]] = getelementptr inbounds [[STRUCT_S:%.*]], ptr [[ARRAY_BEGIN]], i64 5
// CHECK1-NEXT:    br label [[ARRAYDESTROY_BODY:%.*]]
// CHECK1:       arraydestroy.body:
// CHECK1-NEXT:    [[ARRAYDESTROY_ELEMENTPAST:%.*]] = phi ptr [ [[TMP3]], [[ENTRY:%.*]] ], [ [[ARRAYDESTROY_ELEMENT:%.*]], [[ARRAYDESTROY_BODY]] ]
// CHECK1-NEXT:    [[ARRAYDESTROY_ELEMENT]] = getelementptr inbounds [[STRUCT_S]], ptr [[ARRAYDESTROY_ELEMENTPAST]], i64 -1
// CHECK1-NEXT:    call void @_ZN1SD1Ev(ptr nonnull align 4 dereferenceable(4) [[ARRAYDESTROY_ELEMENT]]) #3
// CHECK1-NEXT:    [[ARRAYDESTROY_DONE:%.*]] = icmp eq ptr [[ARRAYDESTROY_ELEMENT]], [[ARRAY_BEGIN]]
// CHECK1-NEXT:    br i1 [[ARRAYDESTROY_DONE]], label [[ARRAYDESTROY_DONE1:%.*]], label [[ARRAYDESTROY_BODY]]
// CHECK1:       arraydestroy.done1:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_ZN1SD1Ev
// CHECK1-SAME: (ptr nonnull align 4 dereferenceable(4) [[THIS:%.*]]) unnamed_addr #[[ATTR1]] align 2 {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    call void @_ZN1SD2Ev(ptr nonnull align 4 dereferenceable(4) [[THIS1]]) #3
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_comb..6
// CHECK1-SAME: (ptr [[TMP0:%.*]], ptr [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[REF_TMP:%.*]] = alloca [[STRUCT_S:%.*]], align 4
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP6:%.*]] = getelementptr [[STRUCT_S]], ptr [[TMP3]], i64 5
// CHECK1-NEXT:    [[OMP_ARRAYCPY_ISEMPTY:%.*]] = icmp eq ptr [[TMP3]], [[TMP6]]
// CHECK1-NEXT:    br i1 [[OMP_ARRAYCPY_ISEMPTY]], label [[OMP_ARRAYCPY_DONE2:%.*]], label [[OMP_ARRAYCPY_BODY:%.*]]
// CHECK1:       omp.arraycpy.body:
// CHECK1-NEXT:    [[OMP_ARRAYCPY_SRCELEMENTPAST:%.*]] = phi ptr [ [[TMP5]], [[ENTRY:%.*]] ], [ [[OMP_ARRAYCPY_SRC_ELEMENT:%.*]], [[OMP_ARRAYCPY_BODY]] ]
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DESTELEMENTPAST:%.*]] = phi ptr [ [[TMP3]], [[ENTRY]] ], [ [[OMP_ARRAYCPY_DEST_ELEMENT:%.*]], [[OMP_ARRAYCPY_BODY]] ]
// CHECK1-NEXT:    call void @_ZplRK1SS1_(ptr dead_on_unwind writable sret([[STRUCT_S]]) align 4 [[REF_TMP]], ptr nonnull align 4 dereferenceable(4) [[OMP_ARRAYCPY_DESTELEMENTPAST]], ptr nonnull align 4 dereferenceable(4) [[OMP_ARRAYCPY_SRCELEMENTPAST]])
// CHECK1-NEXT:    [[CALL:%.*]] = call nonnull align 4 dereferenceable(4) ptr @_ZN1SaSERKS_(ptr nonnull align 4 dereferenceable(4) [[OMP_ARRAYCPY_DESTELEMENTPAST]], ptr nonnull align 4 dereferenceable(4) [[REF_TMP]])
// CHECK1-NEXT:    call void @_ZN1SD1Ev(ptr nonnull align 4 dereferenceable(4) [[REF_TMP]]) #3
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DEST_ELEMENT]] = getelementptr [[STRUCT_S]], ptr [[OMP_ARRAYCPY_DESTELEMENTPAST]], i32 1
// CHECK1-NEXT:    [[OMP_ARRAYCPY_SRC_ELEMENT]] = getelementptr [[STRUCT_S]], ptr [[OMP_ARRAYCPY_SRCELEMENTPAST]], i32 1
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DONE:%.*]] = icmp eq ptr [[OMP_ARRAYCPY_DEST_ELEMENT]], [[TMP6]]
// CHECK1-NEXT:    br i1 [[OMP_ARRAYCPY_DONE]], label [[OMP_ARRAYCPY_DONE2]], label [[OMP_ARRAYCPY_BODY]]
// CHECK1:       omp.arraycpy.done2:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_ZplRK1SS1_
// CHECK1-SAME: (ptr dead_on_unwind noalias writable sret([[STRUCT_S:%.*]]) align 4 [[AGG_RESULT:%.*]], ptr nonnull align 4 dereferenceable(4) [[A:%.*]], ptr nonnull align 4 dereferenceable(4) [[B:%.*]]) #[[ATTR7:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[RESULT_PTR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[A_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[B_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[AGG_RESULT]], ptr [[RESULT_PTR]], align 8
// CHECK1-NEXT:    store ptr [[A]], ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[B]], ptr [[B_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    call void @_ZN1SC1ERKS_(ptr nonnull align 4 dereferenceable(4) [[AGG_RESULT]], ptr nonnull align 4 dereferenceable(4) [[TMP1]])
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_ZN1SaSERKS_
// CHECK1-SAME: (ptr nonnull align 4 dereferenceable(4) [[THIS:%.*]], ptr nonnull align 4 dereferenceable(4) [[TMP0:%.*]]) #[[ATTR7]] align 2 {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    ret ptr [[THIS1]]
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_init..7
// CHECK1-SAME: (ptr noalias [[TMP0:%.*]], ptr noalias [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB1]])
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[TMP5:%.*]] = call ptr @__kmpc_threadprivate_cached(ptr @[[GLOB1]], i32 [[TMP2]], ptr @{{reduction_size[.].+[.]}})
// CHECK1-NEXT:    [[TMP7:%.*]] = load i64, ptr [[TMP5]], align 8
// CHECK1-NEXT:    [[TMP8:%.*]] = getelementptr i16, ptr [[TMP4]], i64 [[TMP7]]
// CHECK1-NEXT:    [[OMP_ARRAYINIT_ISEMPTY:%.*]] = icmp eq ptr [[TMP4]], [[TMP8]]
// CHECK1-NEXT:    br i1 [[OMP_ARRAYINIT_ISEMPTY]], label [[OMP_ARRAYINIT_DONE:%.*]], label [[OMP_ARRAYINIT_BODY:%.*]]
// CHECK1:       omp.arrayinit.body:
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DESTELEMENTPAST:%.*]] = phi ptr [ [[TMP4]], [[ENTRY:%.*]] ], [ [[OMP_ARRAYCPY_DEST_ELEMENT:%.*]], [[OMP_ARRAYINIT_BODY]] ]
// CHECK1-NEXT:    store i16 0, ptr [[OMP_ARRAYCPY_DESTELEMENTPAST]], align 2
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DEST_ELEMENT]] = getelementptr i16, ptr [[OMP_ARRAYCPY_DESTELEMENTPAST]], i32 1
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DONE:%.*]] = icmp eq ptr [[OMP_ARRAYCPY_DEST_ELEMENT]], [[TMP8]]
// CHECK1-NEXT:    br i1 [[OMP_ARRAYCPY_DONE]], label [[OMP_ARRAYINIT_DONE]], label [[OMP_ARRAYINIT_BODY]]
// CHECK1:       omp.arrayinit.done:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.red_comb..8
// CHECK1-SAME: (ptr [[TMP0:%.*]], ptr [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB1]])
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = call ptr @__kmpc_threadprivate_cached(ptr @[[GLOB1]], i32 [[TMP2]], ptr @{{reduction_size[.].+[.]}})
// CHECK1-NEXT:    [[TMP5:%.*]] = load i64, ptr [[TMP3]], align 8
// CHECK1-NEXT:    [[TMP7:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[TMP9:%.*]] = load ptr, ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP10:%.*]] = getelementptr i16, ptr [[TMP7]], i64 [[TMP5]]
// CHECK1-NEXT:    [[OMP_ARRAYCPY_ISEMPTY:%.*]] = icmp eq ptr [[TMP7]], [[TMP10]]
// CHECK1-NEXT:    br i1 [[OMP_ARRAYCPY_ISEMPTY]], label [[OMP_ARRAYCPY_DONE4:%.*]], label [[OMP_ARRAYCPY_BODY:%.*]]
// CHECK1:       omp.arraycpy.body:
// CHECK1-NEXT:    [[OMP_ARRAYCPY_SRCELEMENTPAST:%.*]] = phi ptr [ [[TMP9]], [[ENTRY:%.*]] ], [ [[OMP_ARRAYCPY_SRC_ELEMENT:%.*]], [[OMP_ARRAYCPY_BODY]] ]
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DESTELEMENTPAST:%.*]] = phi ptr [ [[TMP7]], [[ENTRY]] ], [ [[OMP_ARRAYCPY_DEST_ELEMENT:%.*]], [[OMP_ARRAYCPY_BODY]] ]
// CHECK1-NEXT:    [[TMP11:%.*]] = load i16, ptr [[OMP_ARRAYCPY_DESTELEMENTPAST]], align 2
// CHECK1-NEXT:    [[CONV:%.*]] = sext i16 [[TMP11]] to i32
// CHECK1-NEXT:    [[TMP12:%.*]] = load i16, ptr [[OMP_ARRAYCPY_SRCELEMENTPAST]], align 2
// CHECK1-NEXT:    [[CONV2:%.*]] = sext i16 [[TMP12]] to i32
// CHECK1-NEXT:    [[ADD:%.*]] = add nsw i32 [[CONV]], [[CONV2]]
// CHECK1-NEXT:    [[CONV3:%.*]] = trunc i32 [[ADD]] to i16
// CHECK1-NEXT:    store i16 [[CONV3]], ptr [[OMP_ARRAYCPY_DESTELEMENTPAST]], align 2
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DEST_ELEMENT]] = getelementptr i16, ptr [[OMP_ARRAYCPY_DESTELEMENTPAST]], i32 1
// CHECK1-NEXT:    [[OMP_ARRAYCPY_SRC_ELEMENT]] = getelementptr i16, ptr [[OMP_ARRAYCPY_SRCELEMENTPAST]], i32 1
// CHECK1-NEXT:    [[OMP_ARRAYCPY_DONE:%.*]] = icmp eq ptr [[OMP_ARRAYCPY_DEST_ELEMENT]], [[TMP10]]
// CHECK1-NEXT:    br i1 [[OMP_ARRAYCPY_DONE]], label [[OMP_ARRAYCPY_DONE4]], label [[OMP_ARRAYCPY_BODY]]
// CHECK1:       omp.arraycpy.done4:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@main.omp_outlined
// CHECK1-SAME: (ptr noalias [[DOTGLOBAL_TID_:%.*]], ptr noalias [[DOTBOUND_TID_:%.*]], ptr nonnull align 4 dereferenceable(4) [[A:%.*]], i64 [[VLA:%.*]], ptr nonnull align 2 dereferenceable(2) [[D:%.*]], ptr nonnull align 8 dereferenceable(8) [[DOTTASK_RED_:%.*]]) #[[ATTR8:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[A_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[VLA_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[D_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTTASK_RED__ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[AGG_CAPTURED:%.*]] = alloca [[STRUCT_ANON:%.*]], align 8
// CHECK1-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK1-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[DOTBOUND_TID__ADDR]], align 8
// CHECK1-NEXT:    store ptr [[A]], ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[VLA]], ptr [[VLA_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[D]], ptr [[D_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[DOTTASK_RED_]], ptr [[DOTTASK_RED__ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load i64, ptr [[VLA_ADDR]], align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[D_ADDR]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTTASK_RED__ADDR]], align 8
// CHECK1-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP3]], align 8
// CHECK1-NEXT:    [[TMP5:%.*]] = getelementptr inbounds [[STRUCT_ANON]], ptr [[AGG_CAPTURED]], i32 0, i32 0
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[TMP5]], align 8
// CHECK1-NEXT:    [[TMP6:%.*]] = getelementptr inbounds [[STRUCT_ANON]], ptr [[AGG_CAPTURED]], i32 0, i32 1
// CHECK1-NEXT:    store i64 [[TMP1]], ptr [[TMP6]], align 8
// CHECK1-NEXT:    [[TMP7:%.*]] = getelementptr inbounds [[STRUCT_ANON]], ptr [[AGG_CAPTURED]], i32 0, i32 2
// CHECK1-NEXT:    store ptr [[TMP2]], ptr [[TMP7]], align 8
// CHECK1-NEXT:    [[TMP8:%.*]] = getelementptr inbounds [[STRUCT_ANON]], ptr [[AGG_CAPTURED]], i32 0, i32 3
// CHECK1-NEXT:    [[TMP9:%.*]] = load ptr, ptr [[TMP3]], align 8
// CHECK1-NEXT:    store ptr [[TMP9]], ptr [[TMP8]], align 8
// CHECK1-NEXT:    [[TMP10:%.*]] = load ptr, ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK1-NEXT:    [[TMP11:%.*]] = load i32, ptr [[TMP10]], align 4
// CHECK1-NEXT:    [[TMP12:%.*]] = call ptr @__kmpc_omp_task_alloc(ptr @[[GLOB1]], i32 [[TMP11]], i32 1, i64 48, i64 32, ptr @.omp_task_entry.)
// CHECK1-NEXT:    [[TMP14:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T_WITH_PRIVATES:%.*]], ptr [[TMP12]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP15:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T:%.*]], ptr [[TMP14]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP16:%.*]] = load ptr, ptr [[TMP15]], align 8
// CHECK1-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[TMP16]], ptr align 8 [[AGG_CAPTURED]], i64 32, i1 false)
// CHECK1-NEXT:    [[TMP18:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T_WITH_PRIVATES]], ptr [[TMP12]], i32 0, i32 1
// CHECK1-NEXT:    [[TMP20:%.*]] = getelementptr inbounds [[STRUCT__KMP_PRIVATES_T:%.*]], ptr [[TMP18]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP21:%.*]] = load ptr, ptr [[TMP3]], align 8
// CHECK1-NEXT:    store ptr [[TMP21]], ptr [[TMP20]], align 8
// CHECK1-NEXT:    call void @__kmpc_omp_task_begin_if0(ptr @[[GLOB1]], i32 [[TMP11]], ptr [[TMP12]])
// CHECK1-NEXT:    [[TMP22:%.*]] = call i32 @.omp_task_entry.(i32 [[TMP11]], ptr [[TMP12]]) #[[ATTR3]]
// CHECK1-NEXT:    call void @__kmpc_omp_task_complete_if0(ptr @[[GLOB1]], i32 [[TMP11]], ptr [[TMP12]])
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@__omp_offloading_{{.*}}_main_l{{[0-9]+}}
// CHECK1-SAME: (ptr nonnull align 4 dereferenceable(4) [[A:%.*]], i64 [[VLA:%.*]], ptr nonnull align 2 dereferenceable(2) [[D:%.*]], ptr [[DOTTASK_RED_:%.*]]) #[[ATTR9:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[A_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[VLA_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[D_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTTASK_RED__ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK1-NEXT:    store ptr [[A]], ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[VLA]], ptr [[VLA_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[D]], ptr [[D_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[DOTTASK_RED_]], ptr [[DOTTASK_RED__ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load i64, ptr [[VLA_ADDR]], align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[D_ADDR]], align 8
// CHECK1-NEXT:    store i32 0, ptr [[I]], align 4
// CHECK1-NEXT:    br label [[FOR_COND:%.*]]
// CHECK1:       for.cond:
// CHECK1-NEXT:    [[TMP3:%.*]] = load i32, ptr [[I]], align 4
// CHECK1-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP3]], 5
// CHECK1-NEXT:    br i1 [[CMP]], label [[FOR_BODY:%.*]], label [[FOR_END:%.*]]
// CHECK1:       for.body:
// CHECK1-NEXT:    [[TMP4:%.*]] = load i32, ptr [[TMP0]], align 4
// CHECK1-NEXT:    [[IDXPROM_I:%.*]] = sext i32 [[TMP4]] to i64
// CHECK1-NEXT:    [[ARRAYIDX_I:%.*]] = getelementptr inbounds i16, ptr [[TMP2]], i64 [[IDXPROM_I]]
// CHECK1-NEXT:    [[TMP5:%.*]] = load i16, ptr [[ARRAYIDX_I]], align 2
// CHECK1-NEXT:    [[CONV:%.*]] = sext i16 [[TMP5]] to i32
// CHECK1-NEXT:    [[TMP6:%.*]] = load i32, ptr [[TMP0]], align 4
// CHECK1-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP6]], [[CONV]]
// CHECK1-NEXT:    store i32 [[ADD]], ptr [[TMP0]], align 4
// CHECK1-NEXT:    br label [[FOR_INC:%.*]]
// CHECK1:       for.inc:
// CHECK1-NEXT:    [[TMP7:%.*]] = load i32, ptr [[I]], align 4
// CHECK1-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP7]], 1
// CHECK1-NEXT:    store i32 [[INC]], ptr [[I]], align 4
// CHECK1-NEXT:    br label [[FOR_COND]], !llvm.loop [[LOOP3:![0-9]+]]
// CHECK1:       for.end
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.omp_task_privates_map.
// CHECK1-SAME: (ptr noalias [[TMP0:%.*]], ptr noalias [[TMP1:%.*]]) #[[ATTR9:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = getelementptr inbounds [[STRUCT__KMP_PRIVATES_T:%.*]], ptr [[TMP2]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    store ptr [[TMP3]], ptr [[TMP4]], align 8
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@.omp_task_entry.
// CHECK1-SAME: (i32 [[TMP0:%.*]], ptr noalias [[TMP1:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTGLOBAL_TID__ADDR_I:%.*]] = alloca i32, align 4
// CHECK1-NEXT:    [[DOTPART_ID__ADDR_I:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTPRIVATES__ADDR_I:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTCOPY_FN__ADDR_I:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTTASK_T__ADDR_I:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[__CONTEXT_ADDR_I:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTFIRSTPRIV_PTR_ADDR_I:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca i32, align 4
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store i32 [[TMP0]], ptr [[DOTADDR]], align 4
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTADDR]], align 4
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[DOTADDR1]], align 8
// CHECK1-NEXT:    [[TMP4:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T_WITH_PRIVATES:%.*]], ptr [[TMP3]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP5:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T:%.*]], ptr [[TMP4]], i32 0, i32 2
// CHECK1-NEXT:    [[TMP6:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T]], ptr [[TMP4]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP7:%.*]] = load ptr, ptr [[TMP6]], align 8
// CHECK1-NEXT:    [[TMP9:%.*]] = getelementptr inbounds [[STRUCT_KMP_TASK_T_WITH_PRIVATES]], ptr [[TMP3]], i32 0, i32 1
// CHECK1-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META3:![0-9]+]])
// CHECK1-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META8:![0-9]+]])
// CHECK1-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META10:![0-9]+]])
// CHECK1-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META12:![0-9]+]])
// CHECK1-NEXT:    store i32 [[TMP2]], ptr [[DOTGLOBAL_TID__ADDR_I]], align 4, !noalias !14
// CHECK1-NEXT:    store ptr [[TMP5]], ptr [[DOTPART_ID__ADDR_I]], align 8, !noalias !14
// CHECK1-NEXT:    store ptr [[TMP9]], ptr [[DOTPRIVATES__ADDR_I]], align 8, !noalias !14
// CHECK1-NEXT:    store ptr @.omp_task_privates_map., ptr [[DOTCOPY_FN__ADDR_I]], align 8, !noalias !14
// CHECK1-NEXT:    store ptr [[TMP3]], ptr [[DOTTASK_T__ADDR_I]], align 8, !noalias !14
// CHECK1-NEXT:    store ptr [[TMP7]], ptr [[__CONTEXT_ADDR_I]], align 8, !noalias !14
// CHECK1-NEXT:    [[TMP12:%.*]] = load ptr, ptr [[__CONTEXT_ADDR_I]], align 8, !noalias !14
// CHECK1-NEXT:    [[TMP13:%.*]] = getelementptr inbounds [[STRUCT_ANON:%.*]], ptr [[TMP12]], i32 0, i32 1
// CHECK1-NEXT:    [[TMP14:%.*]] = load i64, ptr [[TMP13]], align 8
// CHECK1-NEXT:    [[TMP15:%.*]] = load ptr, ptr [[DOTCOPY_FN__ADDR_I]], align 8, !noalias !14
// CHECK1-NEXT:    [[TMP16:%.*]] = load ptr, ptr [[DOTPRIVATES__ADDR_I]], align 8, !noalias !14
// CHECK1-NEXT:    call void [[TMP15]](ptr [[TMP16]], ptr [[DOTFIRSTPRIV_PTR_ADDR_I]]) #[[ATTR3]]
// CHECK1-NEXT:    [[TMP18:%.*]] = load ptr, ptr [[DOTFIRSTPRIV_PTR_ADDR_I]], align 8, !noalias !14
// CHECK1-NEXT:    [[TMP20:%.*]] = load ptr, ptr [[TMP12]], align 8
// CHECK1-NEXT:    [[TMP21:%.*]] = load ptr, ptr [[TMP18]], align 8
// CHECK1-NEXT:    [[TMP22:%.*]] = load i32, ptr [[DOTGLOBAL_TID__ADDR_I]], align 4, !noalias !14
// CHECK1-NEXT:    [[TMP24:%.*]] = call ptr @__kmpc_task_reduction_get_th_data(i32 [[TMP22]], ptr [[TMP21]], ptr [[TMP20]])
// CHECK1-NEXT:    [[TMP26:%.*]] = load ptr, ptr [[TMP12]], align 8
// CHECK1-NEXT:    [[TMP27:%.*]] = getelementptr inbounds [[STRUCT_ANON]], ptr [[TMP12]], i32 0, i32 2
// CHECK1-NEXT:    [[TMP28:%.*]] = load ptr, ptr [[TMP27]], align 8
// CHECK1-NEXT:    [[TMP29:%.*]] = load ptr, ptr [[TMP18]], align 8
// CHECK1-NEXT:    call void @__omp_offloading_{{.*}}_main_l{{[0-9]+}}(ptr [[TMP26]], i64 [[TMP14]], ptr [[TMP28]], ptr [[TMP29]]) #[[ATTR3]]
// CHECK1-NEXT:    ret i32 0
//
//
// CHECK1-LABEL: define {{[^@]+}}@_ZN1SC2Ev
// CHECK1-SAME: (ptr nonnull align 4 dereferenceable(4) [[THIS:%.*]]) unnamed_addr #[[ATTR1]] align 2 {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    [[A:%.*]] = getelementptr inbounds [[STRUCT_S:%.*]], ptr [[THIS1]], i32 0, i32 0
// CHECK1-NEXT:    store i32 0, ptr [[A]], align 4
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_ZN1SD2Ev
// CHECK1-SAME: (ptr nonnull align 4 dereferenceable(4) [[THIS:%.*]]) unnamed_addr #[[ATTR1]] align 2 {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_ZN1SC1ERKS_
// CHECK1-SAME: (ptr nonnull align 4 dereferenceable(4) [[THIS:%.*]], ptr nonnull align 4 dereferenceable(4) [[TMP0:%.*]]) unnamed_addr #[[ATTR1]] align 2 {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    call void @_ZN1SC2ERKS_(ptr nonnull align 4 dereferenceable(4) [[THIS1]], ptr nonnull align 4 dereferenceable(4) [[TMP1]])
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_ZN1SC2ERKS_
// CHECK1-SAME: (ptr nonnull align 4 dereferenceable(4) [[THIS:%.*]], ptr nonnull align 4 dereferenceable(4) [[TMP0:%.*]]) unnamed_addr #[[ATTR1]] align 2 {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[DOTADDR]], align 8
// CHECK1-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    ret void
//
