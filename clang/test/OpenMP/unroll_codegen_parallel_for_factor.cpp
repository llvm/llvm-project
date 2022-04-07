// Check code generation
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -emit-llvm %s -o - | FileCheck %s --check-prefix=IR

// Check same results after serialization round-trip
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -include-pch %t -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
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
// IR-NEXT:    store i32 %[[START:.+]], i32* %[[START_ADDR]], align 4
// IR-NEXT:    store i32 %[[END:.+]], i32* %[[END_ADDR]], align 4
// IR-NEXT:    store i32 %[[STEP:.+]], i32* %[[STEP_ADDR]], align 4
// IR-NEXT:    call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @2, i32 3, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*, i32*, i32*)* @.omp_outlined. to void (i32*, i32*, ...)*), i32* %[[END_ADDR]], i32* %[[STEP_ADDR]], i32* %[[START_ADDR]])
// IR-NEXT:    ret void
// IR-NEXT:  }
extern "C" void func(int start, int end, int step) {
  #pragma omp parallel for
  #pragma omp unroll partial(7)
  for (int i = start; i < end; i+=step)
    body(start, end, step, i);
}


// IR-LABEL: @.omp_outlined.(
// IR-NEXT:  [[ENTRY:.*]]:
// IR-NEXT:    %[[DOTGLOBAL_TID__ADDR:.+]] = alloca i32*, align 8
// IR-NEXT:    %[[DOTBOUND_TID__ADDR:.+]] = alloca i32*, align 8
// IR-NEXT:    %[[END_ADDR:.+]] = alloca i32*, align 8
// IR-NEXT:    %[[STEP_ADDR:.+]] = alloca i32*, align 8
// IR-NEXT:    %[[START_ADDR:.+]] = alloca i32*, align 8
// IR-NEXT:    %[[DOTOMP_IV:.+]] = alloca i32, align 4
// IR-NEXT:    %[[TMP:.+]] = alloca i32, align 4
// IR-NEXT:    %[[I:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_1:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_2:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_3:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_6:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTCAPTURE_EXPR_8:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTUNROLLED_IV_I:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_LB:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_UB:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_STRIDE:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTOMP_IS_LAST:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTUNROLLED_IV_I12:.+]] = alloca i32, align 4
// IR-NEXT:    %[[DOTUNROLL_INNER_IV_I:.+]] = alloca i32, align 4
// IR-NEXT:    store i32* %[[DOTGLOBAL_TID_:.+]], i32** %[[DOTGLOBAL_TID__ADDR]], align 8
// IR-NEXT:    store i32* %[[DOTBOUND_TID_:.+]], i32** %[[DOTBOUND_TID__ADDR]], align 8
// IR-NEXT:    store i32* %[[END:.+]], i32** %[[END_ADDR]], align 8
// IR-NEXT:    store i32* %[[STEP:.+]], i32** %[[STEP_ADDR]], align 8
// IR-NEXT:    store i32* %[[START:.+]], i32** %[[START_ADDR]], align 8
// IR-NEXT:    %[[TMP0:.+]] = load i32*, i32** %[[END_ADDR]], align 8
// IR-NEXT:    %[[TMP1:.+]] = load i32*, i32** %[[STEP_ADDR]], align 8
// IR-NEXT:    %[[TMP2:.+]] = load i32*, i32** %[[START_ADDR]], align 8
// IR-NEXT:    %[[TMP3:.+]] = load i32, i32* %[[TMP2]], align 4
// IR-NEXT:    store i32 %[[TMP3]], i32* %[[I]], align 4
// IR-NEXT:    %[[TMP4:.+]] = load i32, i32* %[[TMP2]], align 4
// IR-NEXT:    store i32 %[[TMP4]], i32* %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[TMP5:.+]] = load i32, i32* %[[TMP0]], align 4
// IR-NEXT:    store i32 %[[TMP5]], i32* %[[DOTCAPTURE_EXPR_1]], align 4
// IR-NEXT:    %[[TMP6:.+]] = load i32, i32* %[[TMP1]], align 4
// IR-NEXT:    store i32 %[[TMP6]], i32* %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[TMP7:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_1]], align 4
// IR-NEXT:    %[[TMP8:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[SUB:.+]] = sub i32 %[[TMP7]], %[[TMP8]]
// IR-NEXT:    %[[SUB4:.+]] = sub i32 %[[SUB]], 1
// IR-NEXT:    %[[TMP9:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[ADD:.+]] = add i32 %[[SUB4]], %[[TMP9]]
// IR-NEXT:    %[[TMP10:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[DIV:.+]] = udiv i32 %[[ADD]], %[[TMP10]]
// IR-NEXT:    %[[SUB5:.+]] = sub i32 %[[DIV]], 1
// IR-NEXT:    store i32 %[[SUB5]], i32* %[[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    %[[TMP11:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    %[[ADD7:.+]] = add i32 %[[TMP11]], 1
// IR-NEXT:    store i32 %[[ADD7]], i32* %[[DOTCAPTURE_EXPR_6]], align 4
// IR-NEXT:    %[[TMP12:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_6]], align 4
// IR-NEXT:    %[[SUB9:.+]] = sub i32 %[[TMP12]], -6
// IR-NEXT:    %[[DIV10:.+]] = udiv i32 %[[SUB9]], 7
// IR-NEXT:    %[[SUB11:.+]] = sub i32 %[[DIV10]], 1
// IR-NEXT:    store i32 %[[SUB11]], i32* %[[DOTCAPTURE_EXPR_8]], align 4
// IR-NEXT:    store i32 0, i32* %[[DOTUNROLLED_IV_I]], align 4
// IR-NEXT:    %[[TMP13:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_6]], align 4
// IR-NEXT:    %[[CMP:.+]] = icmp ult i32 0, %[[TMP13]]
// IR-NEXT:    br i1 %[[CMP]], label %[[OMP_PRECOND_THEN:.+]], label %[[OMP_PRECOND_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_PRECOND_THEN]]:
// IR-NEXT:    store i32 0, i32* %[[DOTOMP_LB]], align 4
// IR-NEXT:    %[[TMP14:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_8]], align 4
// IR-NEXT:    store i32 %[[TMP14]], i32* %[[DOTOMP_UB]], align 4
// IR-NEXT:    store i32 1, i32* %[[DOTOMP_STRIDE]], align 4
// IR-NEXT:    store i32 0, i32* %[[DOTOMP_IS_LAST]], align 4
// IR-NEXT:    %[[TMP15:.+]] = load i32*, i32** %[[DOTGLOBAL_TID__ADDR]], align 8
// IR-NEXT:    %[[TMP16:.+]] = load i32, i32* %[[TMP15]], align 4
// IR-NEXT:    call void @__kmpc_for_static_init_4u(%struct.ident_t* @1, i32 %[[TMP16]], i32 34, i32* %[[DOTOMP_IS_LAST]], i32* %[[DOTOMP_LB]], i32* %[[DOTOMP_UB]], i32* %[[DOTOMP_STRIDE]], i32 1, i32 1)
// IR-NEXT:    %[[TMP17:.+]] = load i32, i32* %[[DOTOMP_UB]], align 4
// IR-NEXT:    %[[TMP18:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_8]], align 4
// IR-NEXT:    %[[CMP13:.+]] = icmp ugt i32 %[[TMP17]], %[[TMP18]]
// IR-NEXT:    br i1 %[[CMP13]], label %[[COND_TRUE:.+]], label %[[COND_FALSE:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_TRUE]]:
// IR-NEXT:    %[[TMP19:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_8]], align 4
// IR-NEXT:    br label %[[COND_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[COND_FALSE]]:
// IR-NEXT:    %[[TMP20:.+]] = load i32, i32* %[[DOTOMP_UB]], align 4
// IR-NEXT:    br label %[[COND_END]]
// IR-EMPTY:
// IR-NEXT:  [[COND_END]]:
// IR-NEXT:    %[[COND:.+]] = phi i32 [ %[[TMP19]], %[[COND_TRUE]] ], [ %[[TMP20]], %[[COND_FALSE]] ]
// IR-NEXT:    store i32 %[[COND]], i32* %[[DOTOMP_UB]], align 4
// IR-NEXT:    %[[TMP21:.+]] = load i32, i32* %[[DOTOMP_LB]], align 4
// IR-NEXT:    store i32 %[[TMP21]], i32* %[[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_COND]]:
// IR-NEXT:    %[[TMP22:.+]] = load i32, i32* %[[DOTOMP_IV]], align 4
// IR-NEXT:    %[[TMP23:.+]] = load i32, i32* %[[DOTOMP_UB]], align 4
// IR-NEXT:    %[[ADD14:.+]] = add i32 %[[TMP23]], 1
// IR-NEXT:    %[[CMP15:.+]] = icmp ult i32 %[[TMP22]], %[[ADD14]]
// IR-NEXT:    br i1 %[[CMP15]], label %[[OMP_INNER_FOR_BODY:.+]], label %[[OMP_INNER_FOR_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_BODY]]:
// IR-NEXT:    %[[TMP24:.+]] = load i32, i32* %[[DOTOMP_IV]], align 4
// IR-NEXT:    %[[MUL:.+]] = mul i32 %[[TMP24]], 7
// IR-NEXT:    %[[ADD16:.+]] = add i32 0, %[[MUL]]
// IR-NEXT:    store i32 %[[ADD16]], i32* %[[DOTUNROLLED_IV_I12]], align 4
// IR-NEXT:    %[[TMP25:.+]] = load i32, i32* %[[DOTUNROLLED_IV_I12]], align 4
// IR-NEXT:    store i32 %[[TMP25]], i32* %[[DOTUNROLL_INNER_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_COND]]:
// IR-NEXT:    %[[TMP26:.+]] = load i32, i32* %[[DOTUNROLL_INNER_IV_I]], align 4
// IR-NEXT:    %[[TMP27:.+]] = load i32, i32* %[[DOTUNROLLED_IV_I12]], align 4
// IR-NEXT:    %[[ADD17:.+]] = add i32 %[[TMP27]], 7
// IR-NEXT:    %[[CMP18:.+]] = icmp ule i32 %[[TMP26]], %[[ADD17]]
// IR-NEXT:    br i1 %[[CMP18]], label %[[LAND_RHS:.+]], label %[[LAND_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[LAND_RHS]]:
// IR-NEXT:    %[[TMP28:.+]] = load i32, i32* %[[DOTUNROLL_INNER_IV_I]], align 4
// IR-NEXT:    %[[TMP29:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    %[[ADD19:.+]] = add i32 %[[TMP29]], 1
// IR-NEXT:    %[[CMP20:.+]] = icmp ule i32 %[[TMP28]], %[[ADD19]]
// IR-NEXT:    br label %[[LAND_END]]
// IR-EMPTY:
// IR-NEXT:  [[LAND_END]]:
// IR-NEXT:    %[[TMP30:.+]] = phi i1 [ false, %[[FOR_COND]] ], [ %[[CMP20]], %[[LAND_RHS]] ]
// IR-NEXT:    br i1 %[[TMP30]], label %[[FOR_BODY:.+]], label %[[FOR_END:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_BODY]]:
// IR-NEXT:    %[[TMP31:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    %[[TMP32:.+]] = load i32, i32* %[[DOTUNROLL_INNER_IV_I]], align 4
// IR-NEXT:    %[[TMP33:.+]] = load i32, i32* %[[DOTCAPTURE_EXPR_2]], align 4
// IR-NEXT:    %[[MUL21:.+]] = mul i32 %[[TMP32]], %[[TMP33]]
// IR-NEXT:    %[[ADD22:.+]] = add i32 %[[TMP31]], %[[MUL21]]
// IR-NEXT:    store i32 %[[ADD22]], i32* %[[I]], align 4
// IR-NEXT:    %[[TMP34:.+]] = load i32, i32* %[[TMP2]], align 4
// IR-NEXT:    %[[TMP35:.+]] = load i32, i32* %[[TMP0]], align 4
// IR-NEXT:    %[[TMP36:.+]] = load i32, i32* %[[TMP1]], align 4
// IR-NEXT:    %[[TMP37:.+]] = load i32, i32* %[[I]], align 4
// IR-NEXT:    call void (...) @body(i32 noundef %[[TMP34]], i32 noundef %[[TMP35]], i32 noundef %[[TMP36]], i32 noundef %[[TMP37]])
// IR-NEXT:    br label %[[FOR_INC:.+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_INC]]:
// IR-NEXT:    %[[TMP38:.+]] = load i32, i32* %[[DOTUNROLL_INNER_IV_I]], align 4
// IR-NEXT:    %[[INC:.+]] = add i32 %[[TMP38]], 1
// IR-NEXT:    store i32 %[[INC]], i32* %[[DOTUNROLL_INNER_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop ![[LOOP2:[0-9]+]]
// IR-EMPTY:
// IR-NEXT:  [[FOR_END]]:
// IR-NEXT:    br label %[[OMP_BODY_CONTINUE:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_BODY_CONTINUE]]:
// IR-NEXT:    br label %[[OMP_INNER_FOR_INC:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_INC]]:
// IR-NEXT:    %[[TMP39:.+]] = load i32, i32* %[[DOTOMP_IV]], align 4
// IR-NEXT:    %[[ADD23:.+]] = add i32 %[[TMP39]], 1
// IR-NEXT:    store i32 %[[ADD23]], i32* %[[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_INNER_FOR_END]]:
// IR-NEXT:    br label %[[OMP_LOOP_EXIT:.+]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_LOOP_EXIT]]:
// IR-NEXT:    %[[TMP40:.+]] = load i32*, i32** %[[DOTGLOBAL_TID__ADDR]], align 8
// IR-NEXT:    %[[TMP41:.+]] = load i32, i32* %[[TMP40]], align 4
// IR-NEXT:    call void @__kmpc_for_static_fini(%struct.ident_t* @1, i32 %[[TMP41]])
// IR-NEXT:    br label %[[OMP_PRECOND_END]]
// IR-EMPTY:
// IR-NEXT:  [[OMP_PRECOND_END]]:
// IR-NEXT:    ret void
// IR-NEXT:  }

#endif /* HEADER */


// IR: ![[LOOP2]] = distinct !{![[LOOP2]], ![[LOOPPROP3:[0-9]+]], ![[LOOPPROP4:[0-9]+]]}
// IR: ![[LOOPPROP3]] = !{!"llvm.loop.mustprogress"}
// IR: ![[LOOPPROP4]] = !{!"llvm.loop.unroll.count", i32 7}
