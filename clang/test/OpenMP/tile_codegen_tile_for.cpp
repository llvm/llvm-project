// Check code generation
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=IR

// Check same results after serialization round-trip
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -include-pch %t -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

// Account for multiple transformations of a loop before consumed by
// #pragma omp for.

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...) {}


// IR-LABEL: define {{.*}}@func(
// IR-NEXT:  [[ENTRY:.*]]:
// IR-NEXT:    %[[START_ADDR:.+]] = alloca i32, align 4
// IR-NEXT:    %[[END_ADDR:.+]] = alloca i32, align 4
// IR-NEXT:    %[[STEP_ADDR:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_IV:.+]] = alloca i32, align 4
// IR-NEXT:    %[[TMP:.+]] = alloca i32, align 4
// IR-NEXT:    %[[I:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_1:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTNEW_STEP:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_2:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTFLOOR_0_IV_I:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_5:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_7:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_11:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_13:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTFLOOR_0_IV__FLOOR_0_IV_I:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_LB:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_UB:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_STRIDE:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_IS_LAST:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTFLOOR_0_IV__FLOOR_0_IV_I17:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTTILE_0_IV__FLOOR_0_IV_I:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTTILE_0_IV_I:.+]] = alloca i32, align 4
// IR-NEXT:    %[[TMP0:.+]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB2:.+]])
// IR-NEXT:    store i32 %[[START:.+]], ptr %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[END:.+]], ptr %[[END_ADDR]], align 4
// IR-NEXT:    store i32 %[[STEP:.+]], ptr %[[STEP_ADDR]], align 4
// IR-NEXT:    %[[TMP1:.+]] = load i32, ptr %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP1]], ptr %[[I]], align 4
// IR-NEXT:    %[[TMP2:.+]] = load i32, ptr %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP2]], ptr %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[TMP3:.+]] = load i32, ptr %[[END_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP3]], ptr %[[DOTCAPTURE_EXPR_1]], align 4
// IR-NEXT:    %[[TMP4:.+]] = load i32, ptr %[[STEP_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP4]], ptr %[[DOTNEW_STEP]], align 4
// IR-NEXT:    %[[TMP5:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_1]], align 4
// IR-NEXT:    %[[TMP6:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[SUB:.+]] = sub i32 %[[TMP5]], %[[TMP6]]
// IR-NEXT:    %[[SUB3:.+]] = sub i32 %[[SUB]], 1
// IR-NEXT:    %[[TMP7:.+]] = load i32, ptr %[[DOTNEW_STEP]], align 4
// IR-NEXT:    %[[ADD:.+]] = add i32 %[[SUB3]], %[[TMP7]]
// IR-NEXT:    %[[TMP8:.+]] = load i32, ptr %[[DOTNEW_STEP]], align 4
// IR-NEXT:    %[[DIV:.+]] = udiv i32 %[[ADD]], %[[TMP8]]
// IR-NEXT:    %[[SUB4:.+]] = sub i32 %[[DIV]], 1
// IR-NEXT:    store i32 %[[SUB4]], ptr %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    store i32 0, ptr %[[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    %[[TMP9:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[ADD6:.+]] = add i32 %[[TMP9]], 1
// IR-NEXT:    store i32 %[[ADD6]], ptr %[[DOTCAPTURE_EXPR_5]], align 4
// IR-NEXT:    %[[TMP10:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_5]], align 4
// IR-NEXT:    %[[SUB8:.+]] = sub i32 %[[TMP10]], -3
// IR-NEXT:    %[[DIV9:.+]] = udiv i32 %[[SUB8]], 4
// IR-NEXT:    %[[SUB10:.+]] = sub i32 %[[DIV9]], 1
// IR-NEXT:    store i32 %[[SUB10]], ptr %[[DOTCAPTURE_EXPR_7]], align 4
// IR-NEXT:    %[[TMP11:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_7]], align 4
// IR-NEXT:    %[[ADD12:.+]] = add i32 %[[TMP11]], 1
// IR-NEXT:    store i32 %[[ADD12]], ptr %[[DOTCAPTURE_EXPR_11]], align 4
// IR-NEXT:    %[[TMP12:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_11]], align 4
// IR-NEXT:    %[[SUB14:.+]] = sub i32 %[[TMP12]], -2
// IR-NEXT:    %[[DIV15:.+]] = udiv i32 %[[SUB14]], 3
// IR-NEXT:    %[[SUB16:.+]] = sub i32 %[[DIV15]], 1
// IR-NEXT:    store i32 %[[SUB16]], ptr %[[DOTCAPTURE_EXPR_13]], align 4
// IR-NEXT:    store i32 0, ptr %[[DOTFLOOR_0_IV__FLOOR_0_IV_I]], align 4
// IR-NEXT:    %[[TMP13:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_11]], align 4
// IR-NEXT:    %[[CMP:.+]] = icmp ult i32 0, %[[TMP13]]
// IR-NEXT:    br i1 %[[CMP]], label %[[OMP_PRECOND_THEN:.+]], label %[[OMP_PRECOND_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_PRECOND_THEN]]:
// IR-NEXT:    store i32 0, ptr %[[DOTOMP_LB]], align 4
// IR-NEXT:    %[[TMP14:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_13]], align 4
// IR-NEXT:    store i32 %[[TMP14]], ptr %[[DOTOMP_UB]], align 4
// IR-NEXT:    store i32 1, ptr %[[DOTOMP_STRIDE]], align 4
// IR-NEXT:    store i32 0, ptr %[[DOTOMP_IS_LAST]], align 4
// IR-NEXT:    call void @__kmpc_for_static_init_4u(ptr @[[GLOB1:.+]], i32 %[[TMP0]], i32 34, ptr %[[DOTOMP_IS_LAST]], ptr %[[DOTOMP_LB]], ptr %[[DOTOMP_UB]], ptr %[[DOTOMP_STRIDE]], i32 1, i32 1)
// IR-NEXT:    %[[TMP15:.+]] = load i32, ptr %[[DOTOMP_UB]], align 4
// IR-NEXT:    %[[TMP16:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_13]], align 4
// IR-NEXT:    %[[CMP18:.+]] = icmp ugt i32 %[[TMP15]], %[[TMP16]]
// IR-NEXT:    br i1 %[[CMP18]], label %[[COND_TRUE:.+]], label %[[COND_FALSE:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_TRUE]]:
// IR-NEXT:    %[[TMP17:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_13]], align 4
// IR-NEXT:    br label %[[COND_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_FALSE]]:
// IR-NEXT:    %[[TMP18:.+]] = load i32, ptr %[[DOTOMP_UB]], align 4
// IR-NEXT:    br label %[[COND_END]]
// IR-EMPTY:
// IR-NEXT:  [[COND_END]]:
// IR-NEXT:    %[[COND:.+]] = phi i32 [ %[[TMP17]], %[[COND_TRUE]] ], [ %[[TMP18]], %[[COND_FALSE]] ]
// IR-NEXT:    store i32 %[[COND]], ptr %[[DOTOMP_UB]], align 4
// IR-NEXT:    %[[TMP19:.+]] = load i32, ptr %[[DOTOMP_LB]], align 4
// IR-NEXT:    store i32 %[[TMP19]], ptr %[[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_COND]]:
// IR-NEXT:    %[[TMP20:.+]] = load i32, ptr %[[DOTOMP_IV]], align 4
// IR-NEXT:    %[[TMP21:.+]] = load i32, ptr %[[DOTOMP_UB]], align 4
// IR-NEXT:    %[[ADD19:.+]] = add i32 %[[TMP21]], 1
// IR-NEXT:    %[[CMP20:.+]] = icmp ult i32 %[[TMP20]], %[[ADD19]]
// IR-NEXT:    br i1 %[[CMP20]], label %[[OMP_INNER_FOR_BODY:.+]], label %[[OMP_INNER_FOR_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_BODY]]:
// IR-NEXT:    %[[TMP22:.+]] = load i32, ptr %[[DOTOMP_IV]], align 4
// IR-NEXT:    %[[MUL:.+]] = mul i32 %[[TMP22]], 3
// IR-NEXT:    %[[ADD21:.+]] = add i32 0, %[[MUL]]
// IR-NEXT:    store i32 %[[ADD21]], ptr %[[DOTFLOOR_0_IV__FLOOR_0_IV_I17]], align 4
// IR-NEXT:    %[[TMP23:.+]] = load i32, ptr %[[DOTFLOOR_0_IV__FLOOR_0_IV_I17]], align 4
// IR-NEXT:    store i32 %[[TMP23]], ptr %[[DOTTILE_0_IV__FLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_COND]]:
// IR-NEXT:    %[[TMP24:.+]] = load i32, ptr %[[DOTTILE_0_IV__FLOOR_0_IV_I]], align 4
// IR-NEXT:    %[[TMP25:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_7]], align 4
// IR-NEXT:    %[[ADD22:.+]] = add i32 %[[TMP25]], 1
// IR-NEXT:    %[[TMP26:.+]] = load i32, ptr %[[DOTFLOOR_0_IV__FLOOR_0_IV_I17]], align 4
// IR-NEXT:    %[[ADD23:.+]] = add i32 %[[TMP26]], 3
// IR-NEXT:    %[[CMP24:.+]] = icmp ult i32 %[[ADD22]], %[[ADD23]]
// IR-NEXT:    br i1 %[[CMP24]], label %[[COND_TRUE25:.+]], label %[[COND_FALSE27:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_TRUE25]]:
// IR-NEXT:    %[[TMP27:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_7]], align 4
// IR-NEXT:    %[[ADD26:.+]] = add i32 %[[TMP27]], 1
// IR-NEXT:    br label %[[COND_END29:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_FALSE27]]:
// IR-NEXT:    %[[TMP28:.+]] = load i32, ptr %[[DOTFLOOR_0_IV__FLOOR_0_IV_I17]], align 4
// IR-NEXT:    %[[ADD28:.+]] = add i32 %[[TMP28]], 3
// IR-NEXT:    br label %[[COND_END29]]
// IR-EMPTY:
// IR-NEXT:  [[COND_END29]]:
// IR-NEXT:    %[[COND30:.+]] = phi i32 [ %[[ADD26]], %[[COND_TRUE25]] ], [ %[[ADD28]], %[[COND_FALSE27]] ]
// IR-NEXT:    %[[CMP31:.+]] = icmp ult i32 %[[TMP24]], %[[COND30]]
// IR-NEXT:    br i1 %[[CMP31]], label %[[FOR_BODY:.+]], label %[[FOR_END50:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_BODY]]:
// IR-NEXT:    %[[TMP29:.+]] = load i32, ptr %[[DOTTILE_0_IV__FLOOR_0_IV_I]], align 4
// IR-NEXT:    %[[MUL32:.+]] = mul i32 %[[TMP29]], 4
// IR-NEXT:    %[[ADD33:.+]] = add i32 0, %[[MUL32]]
// IR-NEXT:    store i32 %[[ADD33]], ptr %[[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    %[[TMP30:.+]] = load i32, ptr %[[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 %[[TMP30]], ptr %[[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND34:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_COND34]]:
// IR-NEXT:    %[[TMP31:.+]] = load i32, ptr %[[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    %[[TMP32:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[ADD35:.+]] = add i32 %[[TMP32]], 1
// IR-NEXT:    %[[TMP33:.+]] = load i32, ptr %[[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    %[[ADD36:.+]] = add i32 %[[TMP33]], 4
// IR-NEXT:    %[[CMP37:.+]] = icmp ult i32 %[[ADD35]], %[[ADD36]]
// IR-NEXT:    br i1 %[[CMP37]], label %[[COND_TRUE38:.+]], label %[[COND_FALSE40:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_TRUE38]]:
// IR-NEXT:    %[[TMP34:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[ADD39:.+]] = add i32 %[[TMP34]], 1
// IR-NEXT:    br label %[[COND_END42:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_FALSE40]]:
// IR-NEXT:    %[[TMP35:.+]] = load i32, ptr %[[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    %[[ADD41:.+]] = add i32 %[[TMP35]], 4
// IR-NEXT:    br label %[[COND_END42]]
// IR-EMPTY:
// IR-NEXT:  [[COND_END42]]:
// IR-NEXT:    %[[COND43:.+]] = phi i32 [ %[[ADD39]], %[[COND_TRUE38]] ], [ %[[ADD41]], %[[COND_FALSE40]] ]
// IR-NEXT:    %[[CMP44:.+]] = icmp ult i32 %[[TMP31]], %[[COND43]]
// IR-NEXT:    br i1 %[[CMP44]], label %[[FOR_BODY45:.+]], label %[[FOR_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_BODY45]]:
// IR-NEXT:    %[[TMP36:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[TMP37:.+]] = load i32, ptr %[[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    %[[TMP38:.+]] = load i32, ptr %[[DOTNEW_STEP]], align 4
// IR-NEXT:    %[[MUL46:.+]] = mul i32 %[[TMP37]], %[[TMP38]]
// IR-NEXT:    %[[ADD47:.+]] = add i32 %[[TMP36]], %[[MUL46]]
// IR-NEXT:    store i32 %[[ADD47]], ptr %[[I]], align 4
// IR-NEXT:    %[[TMP39:.+]] = load i32, ptr %[[START_ADDR]], align 4
// IR-NEXT:    %[[TMP40:.+]] = load i32, ptr %[[END_ADDR]], align 4
// IR-NEXT:    %[[TMP41:.+]] = load i32, ptr %[[STEP_ADDR]], align 4
// IR-NEXT:    %[[TMP42:.+]] = load i32, ptr %[[I]], align 4
// IR-NEXT:    call void (...) @body(i32 noundef %[[TMP39]], i32 noundef %[[TMP40]], i32 noundef %[[TMP41]], i32 noundef %[[TMP42]])
// IR-NEXT:    br label %[[FOR_INC:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_INC]]:
// IR-NEXT:    %[[TMP43:.+]] = load i32, ptr %[[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    %[[INC:.+]] = add i32 %[[TMP43]], 1
// IR-NEXT:    store i32 %[[INC]], ptr %[[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND34]], !llvm.loop ![[LOOP3:[0-9]+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC48:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_INC48]]:
// IR-NEXT:    %[[TMP44:.+]] = load i32, ptr %[[DOTTILE_0_IV__FLOOR_0_IV_I]], align 4
// IR-NEXT:    %[[INC49:.+]] = add i32 %[[TMP44]], 1
// IR-NEXT:    store i32 %[[INC49]], ptr %[[DOTTILE_0_IV__FLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop ![[LOOP5:[0-9]+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_END50]]:
// IR-NEXT:    br label %[[OMP_BODY_CONTINUE:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_BODY_CONTINUE]]:
// IR-NEXT:    br label %[[OMP_INNER_FOR_INC:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_INC]]:
// IR-NEXT:    %[[TMP45:.+]] = load i32, ptr %[[DOTOMP_IV]], align 4
// IR-NEXT:    %[[ADD51:.+]] = add i32 %[[TMP45]], 1
// IR-NEXT:    store i32 %[[ADD51]], ptr %[[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_END]]:
// IR-NEXT:    br label %[[OMP_LOOP_EXIT:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_LOOP_EXIT]]:
// IR-NEXT:    call void @__kmpc_for_static_fini(ptr @[[GLOB1]], i32 %[[TMP0]])
// IR-NEXT:    br label %[[OMP_PRECOND_END]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_PRECOND_END]]:
// IR-NEXT:    call void @__kmpc_barrier(ptr @[[GLOB3:.+]], i32 %[[TMP0]])
// IR-NEXT:    ret void
// IR-NEXT:  }


extern "C" void func(int start, int end, int step) {
#pragma omp for
#pragma omp tile sizes(3)
#pragma omp tile sizes(4)
  for (int i = start; i < end; i += step)
    body(start, end, step, i);
}

#endif /* HEADER */

// IR: ![[META0:[0-9]+]] = !{i32 1, !"wchar_size", i32 4}
// IR: ![[META1:[0-9]+]] = !{i32 7, !"openmp", i32 51}
// IR: ![[META2:[0-9]+]] =
// IR: ![[LOOP3]] = distinct !{![[LOOP3]], ![[LOOPPROP4:[0-9]+]]}
// IR: ![[LOOPPROP4]] = !{!"llvm.loop.mustprogress"}
// IR: ![[LOOP5]] = distinct !{![[LOOP5]], ![[LOOPPROP4]]}
