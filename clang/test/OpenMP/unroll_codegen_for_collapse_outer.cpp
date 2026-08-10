// Check code generation
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=IR

// Check same results after serialization round-trip
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -include-pch %t -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...) {}


// IR-LABEL: @func(
// IR-NEXT:  [[ENTRY:.*]]:
// IR-NEXT:    %[[START_ADDR:.+]] = alloca i32, align 4
// IR-NEXT:    %[[END_ADDR:.+]] = alloca i32, align 4
// IR-NEXT:    %[[STEP_ADDR:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_IV:.+]] = alloca i64, align 8
// IR-NEXT:    %[[TMP:.+]] = alloca i32, align 4
// IR-NEXT:    %[[TMP1:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_2:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_3:.+]] = alloca i32, align 4
// IR-NEXT:    %[[J:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_4:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_5:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_6:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_7:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_10:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_12:.+]] = alloca i64, align 8
// IR-NEXT:    %[[I:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTUNROLLED_IV_J:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_LB:.+]] = alloca i64, align 8
// IR-NEXT:    %[[DOTOMP_UB:.+]] = alloca i64, align 8
// IR-NEXT:    %[[DOTOMP_STRIDE:.+]] = alloca i64, align 8
// IR-NEXT:    %[[DOTOMP_IS_LAST:.+]] = alloca i32, align 4
// IR-NEXT:    %[[I22:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTUNROLLED_IV_J23:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTUNROLL_INNER_IV_J:.+]] = alloca i32, align 4
// IR-NEXT:    %[[TMP0:.+]] = call i32 @__kmpc_global_thread_num(ptr @2)
// IR-NEXT:    store i32 %[[START:.+]], ptr %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[END:.+]], ptr %[[END_ADDR]], align 4
// IR-NEXT:    store i32 %[[STEP:.+]], ptr %[[STEP_ADDR]], align 4
// IR-NEXT:    %[[TMP1_1:.+]] = load i32, ptr %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP1_1]], ptr %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[TMP2:.+]] = load i32, ptr %[[END_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP2]], ptr %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[TMP3:.+]] = load i32, ptr %[[STEP_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP3]], ptr %[[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    %[[TMP4:.+]] = load i32, ptr %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP4]], ptr %[[J]], align 4
// IR-NEXT:    %[[TMP5:.+]] = load i32, ptr %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP5]], ptr %[[DOTCAPTURE_EXPR_4]], align 4
// IR-NEXT:    %[[TMP6:.+]] = load i32, ptr %[[END_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP6]], ptr %[[DOTCAPTURE_EXPR_5]], align 4
// IR-NEXT:    %[[TMP7:.+]] = load i32, ptr %[[STEP_ADDR]], align 4
// IR-NEXT:    store i32 %[[TMP7]], ptr %[[DOTCAPTURE_EXPR_6]], align 4
// IR-NEXT:    %[[TMP8:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_5]], align 4
// IR-NEXT:    %[[TMP9:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_4]], align 4
// IR-NEXT:    %[[SUB:.+]] = sub i32 %[[TMP8]], %[[TMP9]]
// IR-NEXT:    %[[SUB8:.+]] = sub i32 %[[SUB]], 1
// IR-NEXT:    %[[TMP10:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_6]], align 4
// IR-NEXT:    %[[ADD:.+]] = add i32 %[[SUB8]], %[[TMP10]]
// IR-NEXT:    %[[TMP11:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_6]], align 4
// IR-NEXT:    %[[DIV:.+]] = udiv i32 %[[ADD]], %[[TMP11]]
// IR-NEXT:    %[[SUB9:.+]] = sub i32 %[[DIV]], 1
// IR-NEXT:    store i32 %[[SUB9]], ptr %[[DOTCAPTURE_EXPR_7]], align 4
// IR-NEXT:    %[[TMP12:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_7]], align 4
// IR-NEXT:    %[[ADD11:.+]] = add i32 %[[TMP12]], 1
// IR-NEXT:    store i32 %[[ADD11]], ptr %[[DOTCAPTURE_EXPR_10]], align 4
// IR-NEXT:    %[[TMP13:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[TMP14:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[SUB13:.+]] = sub i32 %[[TMP13]], %[[TMP14]]
// IR-NEXT:    %[[SUB14:.+]] = sub i32 %[[SUB13]], 1
// IR-NEXT:    %[[TMP15:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    %[[ADD15:.+]] = add i32 %[[SUB14]], %[[TMP15]]
// IR-NEXT:    %[[TMP16:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    %[[DIV16:.+]] = udiv i32 %[[ADD15]], %[[TMP16]]
// IR-NEXT:    %[[CONV:.+]] = zext i32 %[[DIV16]] to i64
// IR-NEXT:    %[[TMP17:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_10]], align 4
// IR-NEXT:    %[[SUB17:.+]] = sub i32 %[[TMP17]], -1
// IR-NEXT:    %[[DIV18:.+]] = udiv i32 %[[SUB17]], 2
// IR-NEXT:    %[[CONV19:.+]] = zext i32 %[[DIV18]] to i64
// IR-NEXT:    %[[MUL:.+]] = mul nsw i64 %[[CONV]], %[[CONV19]]
// IR-NEXT:    %[[SUB20:.+]] = sub nsw i64 %[[MUL]], 1
// IR-NEXT:    store i64 %[[SUB20]], ptr %[[DOTCAPTURE_EXPR_12]], align 8
// IR-NEXT:    %[[TMP18:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    store i32 %[[TMP18]], ptr %[[I]], align 4
// IR-NEXT:    store i32 0, ptr %[[DOTUNROLLED_IV_J]], align 4
// IR-NEXT:    %[[TMP19:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[TMP20:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[CMP:.+]] = icmp slt i32 %[[TMP19]], %[[TMP20]]
// IR-NEXT:    br i1 %[[CMP]], label %[[LAND_LHS_TRUE:.+]], label %[[OMP_PRECOND_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[LAND_LHS_TRUE]]:
// IR-NEXT:    %[[TMP21:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_10]], align 4
// IR-NEXT:    %[[CMP21:.+]] = icmp ult i32 0, %[[TMP21]]
// IR-NEXT:    br i1 %[[CMP21]], label %[[OMP_PRECOND_THEN:.+]], label %[[OMP_PRECOND_END]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_PRECOND_THEN]]:
// IR-NEXT:    store i64 0, ptr %[[DOTOMP_LB]], align 8
// IR-NEXT:    %[[TMP22:.+]] = load i64, ptr %[[DOTCAPTURE_EXPR_12]], align 8
// IR-NEXT:    store i64 %[[TMP22]], ptr %[[DOTOMP_UB]], align 8
// IR-NEXT:    store i64 1, ptr %[[DOTOMP_STRIDE]], align 8
// IR-NEXT:    store i32 0, ptr %[[DOTOMP_IS_LAST]], align 4
// IR-NEXT:    call void @__kmpc_for_static_init_8(ptr @1, i32 %[[TMP0]], i32 34, ptr %[[DOTOMP_IS_LAST]], ptr %[[DOTOMP_LB]], ptr %[[DOTOMP_UB]], ptr %[[DOTOMP_STRIDE]], i64 1, i64 1)
// IR-NEXT:    %[[TMP23:.+]] = load i64, ptr %[[DOTOMP_UB]], align 8
// IR-NEXT:    %[[TMP24:.+]] = load i64, ptr %[[DOTCAPTURE_EXPR_12]], align 8
// IR-NEXT:    %[[CMP24:.+]] = icmp sgt i64 %[[TMP23]], %[[TMP24]]
// IR-NEXT:    br i1 %[[CMP24]], label %[[COND_TRUE:.+]], label %[[COND_FALSE:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_TRUE]]:
// IR-NEXT:    %[[TMP25:.+]] = load i64, ptr %[[DOTCAPTURE_EXPR_12]], align 8
// IR-NEXT:    br label %[[COND_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_FALSE]]:
// IR-NEXT:    %[[TMP26:.+]] = load i64, ptr %[[DOTOMP_UB]], align 8
// IR-NEXT:    br label %[[COND_END]]
// IR-EMPTY:
// IR-NEXT:  [[COND_END]]:
// IR-NEXT:    %[[COND:.+]] = phi i64 [ %[[TMP25]], %[[COND_TRUE]] ], [ %[[TMP26]], %[[COND_FALSE]] ]
// IR-NEXT:    store i64 %[[COND]], ptr %[[DOTOMP_UB]], align 8
// IR-NEXT:    %[[TMP27:.+]] = load i64, ptr %[[DOTOMP_LB]], align 8
// IR-NEXT:    store i64 %[[TMP27]], ptr %[[DOTOMP_IV]], align 8
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_COND]]:
// IR-NEXT:    %[[TMP28:.+]] = load i64, ptr %[[DOTOMP_IV]], align 8
// IR-NEXT:    %[[TMP29:.+]] = load i64, ptr %[[DOTOMP_UB]], align 8
// IR-NEXT:    %[[CMP25:.+]] = icmp sle i64 %[[TMP28]], %[[TMP29]]
// IR-NEXT:    br i1 %[[CMP25]], label %[[OMP_INNER_FOR_BODY:.+]], label %[[OMP_INNER_FOR_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_BODY]]:
// IR-NEXT:    %[[TMP30:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[CONV26:.+]] = sext i32 %[[TMP30]] to i64
// IR-NEXT:    %[[TMP31:.+]] = load i64, ptr %[[DOTOMP_IV]], align 8
// IR-NEXT:    %[[TMP32:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_10]], align 4
// IR-NEXT:    %[[SUB27:.+]] = sub i32 %[[TMP32]], -1
// IR-NEXT:    %[[DIV28:.+]] = udiv i32 %[[SUB27]], 2
// IR-NEXT:    %[[MUL29:.+]] = mul i32 1, %[[DIV28]]
// IR-NEXT:    %[[CONV30:.+]] = zext i32 %[[MUL29]] to i64
// IR-NEXT:    %[[DIV31:.+]] = sdiv i64 %[[TMP31]], %[[CONV30]]
// IR-NEXT:    %[[TMP33:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    %[[CONV32:.+]] = sext i32 %[[TMP33]] to i64
// IR-NEXT:    %[[MUL33:.+]] = mul nsw i64 %[[DIV31]], %[[CONV32]]
// IR-NEXT:    %[[ADD34:.+]] = add nsw i64 %[[CONV26]], %[[MUL33]]
// IR-NEXT:    %[[CONV35:.+]] = trunc i64 %[[ADD34]] to i32
// IR-NEXT:    store i32 %[[CONV35]], ptr %[[I22]], align 4
// IR-NEXT:    %[[TMP34:.+]] = load i64, ptr %[[DOTOMP_IV]], align 8
// IR-NEXT:    %[[TMP35:.+]] = load i64, ptr %[[DOTOMP_IV]], align 8
// IR-NEXT:    %[[TMP36:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_10]], align 4
// IR-NEXT:    %[[SUB36:.+]] = sub i32 %[[TMP36]], -1
// IR-NEXT:    %[[DIV37:.+]] = udiv i32 %[[SUB36]], 2
// IR-NEXT:    %[[MUL38:.+]] = mul i32 1, %[[DIV37]]
// IR-NEXT:    %[[CONV39:.+]] = zext i32 %[[MUL38]] to i64
// IR-NEXT:    %[[DIV40:.+]] = sdiv i64 %[[TMP35]], %[[CONV39]]
// IR-NEXT:    %[[TMP37:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_10]], align 4
// IR-NEXT:    %[[SUB41:.+]] = sub i32 %[[TMP37]], -1
// IR-NEXT:    %[[DIV42:.+]] = udiv i32 %[[SUB41]], 2
// IR-NEXT:    %[[MUL43:.+]] = mul i32 1, %[[DIV42]]
// IR-NEXT:    %[[CONV44:.+]] = zext i32 %[[MUL43]] to i64
// IR-NEXT:    %[[MUL45:.+]] = mul nsw i64 %[[DIV40]], %[[CONV44]]
// IR-NEXT:    %[[SUB46:.+]] = sub nsw i64 %[[TMP34]], %[[MUL45]]
// IR-NEXT:    %[[MUL47:.+]] = mul nsw i64 %[[SUB46]], 2
// IR-NEXT:    %[[ADD48:.+]] = add nsw i64 0, %[[MUL47]]
// IR-NEXT:    %[[CONV49:.+]] = trunc i64 %[[ADD48]] to i32
// IR-NEXT:    store i32 %[[CONV49]], ptr %[[DOTUNROLLED_IV_J23]], align 4
// IR-NEXT:    %[[TMP38:.+]] = load i32, ptr %[[DOTUNROLLED_IV_J23]], align 4
// IR-NEXT:    store i32 %[[TMP38]], ptr %[[DOTUNROLL_INNER_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_COND]]:
// IR-NEXT:    %[[TMP39:.+]] = load i32, ptr %[[DOTUNROLL_INNER_IV_J]], align 4
// IR-NEXT:    %[[TMP40:.+]] = load i32, ptr %[[DOTUNROLLED_IV_J23]], align 4
// IR-NEXT:    %[[ADD50:.+]] = add i32 %[[TMP40]], 2
// IR-NEXT:    %[[CMP51:.+]] = icmp ult i32 %[[TMP39]], %[[ADD50]]
// IR-NEXT:    br i1 %[[CMP51]], label %[[LAND_RHS:.+]], label %[[LAND_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[LAND_RHS]]:
// IR-NEXT:    %[[TMP41:.+]] = load i32, ptr %[[DOTUNROLL_INNER_IV_J]], align 4
// IR-NEXT:    %[[TMP42:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_7]], align 4
// IR-NEXT:    %[[ADD52:.+]] = add i32 %[[TMP42]], 1
// IR-NEXT:    %[[CMP53:.+]] = icmp ult i32 %[[TMP41]], %[[ADD52]]
// IR-NEXT:    br label %[[LAND_END]]
// IR-EMPTY:
// IR-NEXT:  [[LAND_END]]:
// IR-NEXT:    %[[TMP43:.+]] = phi i1 [ false, %[[FOR_COND]] ], [ %[[CMP53]], %[[LAND_RHS]] ]
// IR-NEXT:    br i1 %[[TMP43]], label %[[FOR_BODY:.+]], label %[[FOR_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_BODY]]:
// IR-NEXT:    %[[TMP44:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_4]], align 4
// IR-NEXT:    %[[TMP45:.+]] = load i32, ptr %[[DOTUNROLL_INNER_IV_J]], align 4
// IR-NEXT:    %[[TMP46:.+]] = load i32, ptr %[[DOTCAPTURE_EXPR_6]], align 4
// IR-NEXT:    %[[MUL54:.+]] = mul i32 %[[TMP45]], %[[TMP46]]
// IR-NEXT:    %[[ADD55:.+]] = add i32 %[[TMP44]], %[[MUL54]]
// IR-NEXT:    store i32 %[[ADD55]], ptr %[[J]], align 4
// IR-NEXT:    %[[TMP47:.+]] = load i32, ptr %[[START_ADDR]], align 4
// IR-NEXT:    %[[TMP48:.+]] = load i32, ptr %[[END_ADDR]], align 4
// IR-NEXT:    %[[TMP49:.+]] = load i32, ptr %[[STEP_ADDR]], align 4
// IR-NEXT:    %[[TMP50:.+]] = load i32, ptr %[[I22]], align 4
// IR-NEXT:    %[[TMP51:.+]] = load i32, ptr %[[J]], align 4
// IR-NEXT:    call void (...) @body(i32 noundef %[[TMP47]], i32 noundef %[[TMP48]], i32 noundef %[[TMP49]], i32 noundef %[[TMP50]], i32 noundef %[[TMP51]])
// IR-NEXT:    br label %[[FOR_INC:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_INC]]:
// IR-NEXT:    %[[TMP52:.+]] = load i32, ptr %[[DOTUNROLL_INNER_IV_J]], align 4
// IR-NEXT:    %[[INC:.+]] = add i32 %[[TMP52]], 1
// IR-NEXT:    store i32 %[[INC]], ptr %[[DOTUNROLL_INNER_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop ![[LOOP2:[0-9]+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_END]]:
// IR-NEXT:    br label %[[OMP_BODY_CONTINUE:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_BODY_CONTINUE]]:
// IR-NEXT:    br label %[[OMP_INNER_FOR_INC:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_INC]]:
// IR-NEXT:    %[[TMP53:.+]] = load i64, ptr %[[DOTOMP_IV]], align 8
// IR-NEXT:    %[[ADD56:.+]] = add nsw i64 %[[TMP53]], 1
// IR-NEXT:    store i64 %[[ADD56]], ptr %[[DOTOMP_IV]], align 8
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_END]]:
// IR-NEXT:    br label %[[OMP_LOOP_EXIT:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_LOOP_EXIT]]:
// IR-NEXT:    call void @__kmpc_for_static_fini(ptr @1, i32 %[[TMP0]])
// IR-NEXT:    br label %[[OMP_PRECOND_END]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_PRECOND_END]]:
// IR-NEXT:    call void @__kmpc_barrier(ptr @3, i32 %[[TMP0]])
// IR-NEXT:    ret void
// IR-NEXT:  }
extern "C" void func(int start, int end, int step) {
  #pragma omp for collapse(2)
  for (int i = start; i < end; i+=step) {
    #pragma omp unroll partial
    for (int j = start; j < end; j+=step)
        body(start, end, step, i, j);
  }
}

#endif /* HEADER */


// IR: ![[LOOP2]] = distinct !{![[LOOP2]], ![[LOOPPROP3:[0-9]+]], ![[LOOPPROP4:[0-9]+]]}
// IR: ![[LOOPPROP3]] = !{!"llvm.loop.mustprogress"}
// IR: ![[LOOPPROP4]] = !{!"llvm.loop.unroll.count", i32 2}
