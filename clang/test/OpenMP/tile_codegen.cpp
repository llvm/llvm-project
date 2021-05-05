// Check code generation
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -emit-llvm %s -o - | FileCheck %s --check-prefix=IR

// Check same results after serialization round-trip
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -include-pch %t -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...) {}

// IR: define {{.*}}void @_ZN1SC2Ev(%struct.S*
// IR:         [[THIS_ADDR:%.+]] = alloca %struct.S*, align 8
// IR-NEXT:    [[I_REF:%.+]] = alloca i32*, align 8
// IR-NEXT:    [[FLOOR:%.+]] = alloca i32, align 4
// IR-NEXT:    [[TILE:%.+]] = alloca i32, align 4
// IR-NEXT:    store %struct.S* %{{.+}}, %struct.S** [[THIS_ADDR]], align 8
// IR-NEXT:    [[THIS:%.+]] = load %struct.S*, %struct.S** [[THIS_ADDR]], align 8
// IR-NEXT:    [[I:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[THIS]], i32 0, i32 0
// IR-NEXT:    store i32* [[I]], i32** [[I_REF]], align 8
// IR-NEXT:    store i32 0, i32* [[FLOOR]], align 4
// IR-NEXT:    br label %[[FOR_COND:.+]]
// IR:         [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.+]] = load i32, i32* [[FLOOR]], align 4
// IR-NEXT:    [[CMP:%.+]] = icmp slt i32 [[TMP0]], 4
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.+]], label %[[FOR_END11:.+]]
// IR:         [[FOR_BODY]]:
// IR-NEXT:    [[TMP1:%.+]] = load i32, i32* [[FLOOR]], align 4
// IR-NEXT:    store i32 [[TMP1]], i32* [[TILE]], align 4
// IR-NEXT:    br label %[[FOR_COND3:.+]]
// IR:         [[FOR_COND3]]:
// IR-NEXT:    [[TMP2:%.+]] = load i32, i32* [[TILE]], align 4
// IR-NEXT:    [[TMP3:%.+]] = load i32, i32* [[FLOOR]], align 4
// IR-NEXT:    [[ADD:%.+]] = add nsw i32 [[TMP3]], 5
// IR-NEXT:    [[CMP4:%.+]] = icmp slt i32 4, [[ADD]]
// IR-NEXT:    br i1 [[CMP4]], label %[[COND_TRUE:.+]], label %[[COND_FALSE:.+]]
// IR:         [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.+]]
// IR:         [[COND_FALSE]]:
// IR-NEXT:    [[TMP4:%.+]] = load i32, i32* [[FLOOR]], align 4
// IR-NEXT:    [[ADD5:%.+]] = add nsw i32 [[TMP4]], 5
// IR-NEXT:    br label %[[COND_END]]
// IR:         [[COND_END]]:
// IR-NEXT:    [[COND:%.+]] = phi i32 [ 4, %[[COND_TRUE]] ], [ [[ADD5]], %[[COND_FALSE]] ]
// IR-NEXT:    [[CMP6:%.+]] = icmp slt i32 [[TMP2]], [[COND]]
// IR-NEXT:    br i1 [[CMP6]], label %[[FOR_BODY7:.+]], label %[[FOR_END:.+]]
// IR:         [[FOR_BODY7]]:
// IR-NEXT:    [[TMP5:%.+]] = load i32, i32* [[TILE]], align 4
// IR-NEXT:    [[MUL:%.+]] = mul nsw i32 [[TMP5]], 3
// IR-NEXT:    [[ADD8:%.+]] = add nsw i32 7, [[MUL]]
// IR-NEXT:    [[TMP6:%.+]] = load i32*, i32** [[I_REF]], align 8
// IR-NEXT:    store i32 [[ADD8]], i32* [[TMP6]], align 4
// IR-NEXT:    [[TMP7:%.+]] = load i32*, i32** [[I_REF]], align 8
// IR-NEXT:    [[TMP8:%.+]] = load i32, i32* [[TMP7]], align 4
// IR-NEXT:    call void (...) @body(i32 [[TMP8]])
// IR-NEXT:    br label %[[FOR_INC:.+]]
// IR:         [[FOR_INC]]:
// IR-NEXT:    [[TMP9:%.+]] = load i32, i32* [[TILE]], align 4
// IR-NEXT:    [[INC:%.+]] = add nsw i32 [[TMP9]], 1
// IR-NEXT:    store i32 [[INC]], i32* [[TILE]], align 4
// IR-NEXT:    br label %[[FOR_COND3]]
// IR:         [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC9:.+]]
// IR:         [[FOR_INC9]]:
// IR-NEXT:    [[TMP10:%.+]] = load i32, i32* [[FLOOR]], align 4
// IR-NEXT:    [[ADD10:%.+]] = add nsw i32 [[TMP10]], 5
// IR-NEXT:    store i32 [[ADD10]], i32* [[FLOOR]], align 4
// IR-NEXT:    br label %[[FOR_COND]]
// IR:         [[FOR_END11]]:
// IR-NEXT:    ret void

struct S {
  int i;
  S() {
#pragma omp tile sizes(5)
    for (i = 7; i < 17; i += 3)
      body(i);
  }
} s;

// IR-LABEL: define {{.*}}void @foo1(
// IR:         [[START_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[END_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[STEP_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[CAP_EXPR:%.+]] = alloca i32, align 4
// IR-NEXT:    [[CAP_EXPR1:%.+]] = alloca i32, align 4
// IR-NEXT:    [[CAP_EXPR2:%.+]] = alloca i32, align 4
// IR-NEXT:    [[CAP_EXPR3:%.+]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32 [[START:%.*]], i32* [[START_ADDR]], align 4
// IR-NEXT:    store i32 [[END:%.*]], i32* [[END_ADDR]], align 4
// IR-NEXT:    store i32 [[STEP:%.*]], i32* [[STEP_ADDR]], align 4
// IR-NEXT:    [[TMP0:%.+]] = load i32, i32* [[START_ADDR]], align 4
// IR-NEXT:    store i32 [[TMP0]], i32* [[CAP_EXPR]], align 4
// IR-NEXT:    [[TMP1:%.+]] = load i32, i32* [[END_ADDR]], align 4
// IR-NEXT:    store i32 [[TMP1]], i32* [[CAP_EXPR1]], align 4
// IR-NEXT:    [[TMP2:%.+]] = load i32, i32* [[STEP_ADDR]], align 4
// IR-NEXT:    store i32 [[TMP2]], i32* [[CAP_EXPR2]], align 4
// IR-NEXT:    [[TMP3:%.+]] = load i32, i32* [[CAP_EXPR1]], align 4
// IR-NEXT:    [[TMP4:%.+]] = load i32, i32* [[CAP_EXPR]], align 4
// IR-NEXT:    [[SUB:%.+]] = sub i32 [[TMP3]], [[TMP4]]
// IR-NEXT:    [[SUB4:%.+]] = sub i32 [[SUB]], 1
// IR-NEXT:    [[TMP5:%.+]] = load i32, i32* [[CAP_EXPR2]], align 4
// IR-NEXT:    [[ADD:%.+]] = add i32 [[SUB4]], [[TMP5]]
// IR-NEXT:    [[TMP6:%.+]] = load i32, i32* [[CAP_EXPR2]], align 4
// IR-NEXT:    [[DIV:%.+]] = udiv i32 [[ADD]], [[TMP6]]
// IR-NEXT:    [[SUB5:%.+]] = sub i32 [[DIV]], 1
// IR-NEXT:    store i32 [[SUB5]], i32* [[CAP_EXPR3]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:         [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[TMP8:%.+]] = load i32, i32* [[CAP_EXPR3]], align 4
// IR-NEXT:    [[ADD3:%.*]] = add i32 [[TMP8]], 1
// IR-NEXT:    [[CMP:%.*]] = icmp ult i32 [[TMP0]], [[ADD3]]
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END25:.*]]
// IR:         [[FOR_BODY]]:
// IR-NEXT:    [[TMP5:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP5]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4:.*]]
// IR:         [[FOR_COND4]]:
// IR-NEXT:    [[TMP6:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP11:%.+]] = load i32, i32* [[CAP_EXPR3]], align 4
// IR-NEXT:    [[ADD10:%.*]] = add i32 [[TMP11]], 1
// IR-NEXT:    [[TMP11:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD11:%.*]] = add nsw i32 [[TMP11]], 5
// IR-NEXT:    [[CMP12:%.*]] = icmp ult i32 [[ADD10]], [[ADD11]]
// IR-NEXT:    br i1 [[CMP12]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:         [[COND_TRUE]]:
// IR-NEXT:    [[TMP13:%.+]] = load i32, i32* [[CAP_EXPR3]], align 4
// IR-NEXT:    [[ADD18:%.*]] = add i32 [[TMP13]], 1
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:         [[COND_FALSE]]:
// IR-NEXT:    [[TMP16:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD19:%.*]] = add nsw i32 [[TMP16]], 5
// IR-NEXT:    br label %[[COND_END]]
// IR:         [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ [[ADD18]], %[[COND_TRUE]] ], [ [[ADD19]], %[[COND_FALSE]] ]
// IR-NEXT:    [[CMP20:%.*]] = icmp ult i32 [[TMP6]], [[COND]]
// IR-NEXT:    br i1 [[CMP20]], label %[[FOR_BODY21:.*]], label %[[FOR_END:.*]]
// IR:         [[FOR_BODY21]]:
// IR-NEXT:    [[TMP15:%.+]] = load i32, i32* [[CAP_EXPR]], align 4
// IR-NEXT:    [[TMP19:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP17:%.+]] = load i32, i32* [[CAP_EXPR2]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul i32 [[TMP19]], [[TMP17]]
// IR-NEXT:    [[ADD22:%.*]] = add i32 [[TMP15]], [[MUL]]
// IR-NEXT:    store i32 [[ADD22]], i32* [[I]], align 4
// IR-NEXT:    [[TMP21:%.*]] = load i32, i32* [[I]], align 4
// IR-NEXT:    call void (...) @body(i32 [[TMP21]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:         [[FOR_INC]]:
// IR-NEXT:    [[TMP22:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP22]], 1
// IR-NEXT:    store i32 [[INC]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4]]
// IR:         [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC23:.*]]
// IR:         [[FOR_INC23]]:
// IR-NEXT:    [[TMP23:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD24:%.*]] = add nsw i32 [[TMP23]], 5
// IR-NEXT:    store i32 [[ADD24]], i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]]
// IR:        [[FOR_END25]]:
// IR-NEXT:    ret void
//
extern "C" void foo1(int start, int end, int step) {
  int i;
#pragma omp tile sizes(5)
  for (i = start; i < end; i += step)
    body(i);
}

// IR-LABEL: define {{.*}}void @foo2(
// IR-NEXT:  entry:
// IR-NEXT:    [[START_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[END_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[STEP_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_1_IV_J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_1_IV_J:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32 [[START:%.*]], i32* [[START_ADDR]], align 4
// IR-NEXT:    store i32 [[END:%.*]], i32* [[END_ADDR]], align 4
// IR-NEXT:    store i32 [[STEP:%.*]], i32* [[STEP_ADDR]], align 4
// IR-NEXT:    store i32 7, i32* [[I]], align 4
// IR-NEXT:    store i32 7, i32* [[J]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:         [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 4
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END30:.*]]
// IR:         [[FOR_BODY]]:
// IR-NEXT:    store i32 0, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND1:.*]]
// IR:         [[FOR_COND1]]:
// IR-NEXT:    [[TMP1:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[CMP2:%.*]] = icmp slt i32 [[TMP1]], 4
// IR-NEXT:    br i1 [[CMP2]], label %[[FOR_BODY3:.*]], label %[[FOR_END27:.*]]
// IR:         [[FOR_BODY3]]:
// IR-NEXT:    [[TMP2:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP2]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4:.*]]
// IR:         [[FOR_COND4]]:
// IR-NEXT:    [[TMP3:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP4:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP4]], 5
// IR-NEXT:    [[CMP5:%.*]] = icmp slt i32 4, [[ADD]]
// IR-NEXT:    br i1 [[CMP5]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:         [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:         [[COND_FALSE]]:
// IR-NEXT:    [[TMP5:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD6:%.*]] = add nsw i32 [[TMP5]], 5
// IR-NEXT:    br label %[[COND_END]]
// IR:         [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 4, %[[COND_TRUE]] ], [ [[ADD6]], %[[COND_FALSE]] ]
// IR-NEXT:    [[CMP7:%.*]] = icmp slt i32 [[TMP3]], [[COND]]
// IR-NEXT:    br i1 [[CMP7]], label %[[FOR_BODY8:.*]], label %[[FOR_END24:.*]]
// IR:         [[FOR_BODY8]]:
// IR-NEXT:    [[TMP6:%.+]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL:%.+]] = mul nsw i32 [[TMP6]], 3
// IR-NEXT:    [[ADD9:%.+]] = add nsw i32 7, [[MUL]]
// IR-NEXT:    store i32 [[ADD9]], i32* [[I]], align 4
// IR-NEXT:    [[TMP7:%.+]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    store i32 [[TMP7]], i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND10:.+]]
// IR:         [[FOR_COND10]]:
// IR-NEXT:    [[TMP7:%.*]] = load i32, i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[TMP8:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD10:%.*]] = add nsw i32 [[TMP8]], 5
// IR-NEXT:    [[CMP11:%.*]] = icmp slt i32 4, [[ADD10]]
// IR-NEXT:    br i1 [[CMP11]], label %[[COND_TRUE12:.*]], label %[[COND_FALSE13:.*]]
// IR:         [[COND_TRUE12]]:
// IR-NEXT:    br label %[[COND_END15:.*]]
// IR:         [[COND_FALSE13]]:
// IR-NEXT:    [[TMP9:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD14:%.*]] = add nsw i32 [[TMP9]], 5
// IR-NEXT:    br label %[[COND_END15]]
// IR:         [[COND_END15]]:
// IR-NEXT:    [[COND16:%.*]] = phi i32 [ 4, %[[COND_TRUE12]] ], [ [[ADD14]], %[[COND_FALSE13]] ]
// IR-NEXT:    [[CMP17:%.*]] = icmp slt i32 [[TMP7]], [[COND16]]
// IR-NEXT:    br i1 [[CMP17]], label %[[FOR_BODY18:.*]], label %[[FOR_END:.*]]
// IR:         [[FOR_BODY18]]:
// IR-NEXT:    [[TMP11:%.*]] = load i32, i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[MUL20:%.*]] = mul nsw i32 [[TMP11]], 3
// IR-NEXT:    [[ADD21:%.*]] = add nsw i32 7, [[MUL20]]
// IR-NEXT:    store i32 [[ADD21]], i32* [[J]], align 4
// IR-NEXT:    [[TMP12:%.*]] = load i32, i32* [[I]], align 4
// IR-NEXT:    [[TMP13:%.*]] = load i32, i32* [[J]], align 4
// IR-NEXT:    call void (...) @body(i32 [[TMP12]], i32 [[TMP13]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:         [[FOR_INC]]:
// IR-NEXT:    [[TMP14:%.*]] = load i32, i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP14]], 1
// IR-NEXT:    store i32 [[INC]], i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND10]]
// IR:         [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC22:.*]]
// IR:         [[FOR_INC22]]:
// IR-NEXT:    [[TMP15:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC23:%.*]] = add nsw i32 [[TMP15]], 1
// IR-NEXT:    store i32 [[INC23]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4]]
// IR:         [[FOR_END24]]:
// IR-NEXT:    br label %[[FOR_INC25:.*]]
// IR:         [[FOR_INC25]]:
// IR-NEXT:    [[TMP16:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD26:%.*]] = add nsw i32 [[TMP16]], 5
// IR-NEXT:    store i32 [[ADD26]], i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND1]]
// IR:         [[FOR_END27]]:
// IR-NEXT:    br label %[[FOR_INC28:.*]]
// IR:         [[FOR_INC28]]:
// IR-NEXT:    [[TMP17:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD29:%.*]] = add nsw i32 [[TMP17]], 5
// IR-NEXT:    store i32 [[ADD29]], i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]]
// IR:         [[FOR_END30]]:
// IR-NEXT:    ret void
//
extern "C" void foo2(int start, int end, int step) {
#pragma omp tile sizes(5,5)
  for (int i = 7; i < 17; i+=3)
    for (int j = 7; j < 17; j+=3)
      body(i,j);
}

// IR-LABEL: @foo3(
// IR-NEXT:  entry:
// IR-NEXT:    [[DOTOMP_IV:%.*]] = alloca i32, align 4
// IR-NEXT:    [[TMP:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_LB:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_UB:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_STRIDE:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_1_IV_J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_1_IV_J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[GLOB2:@.*]])
// IR-NEXT:    store i32 0, i32* [[DOTOMP_LB]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    store i32 1, i32* [[DOTOMP_STRIDE]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTOMP_IS_LAST]], align 4
// IR-NEXT:    call void @__kmpc_for_static_init_4(%struct.ident_t* [[GLOB1:@.*]], i32 [[TMP0]], i32 34, i32* [[DOTOMP_IS_LAST]], i32* [[DOTOMP_LB]], i32* [[DOTOMP_UB]], i32* [[DOTOMP_STRIDE]], i32 1, i32 1)
// IR-NEXT:    [[TMP1:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[TMP1]], 0
// IR-NEXT:    br i1 [[CMP]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:         [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:         [[COND_FALSE]]:
// IR-NEXT:    [[TMP2:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    br label %[[COND_END]]
// IR:         [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 0, %[[COND_TRUE]] ], [ [[TMP2]], %[[COND_FALSE]] ]
// IR-NEXT:    store i32 [[COND]], i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    [[TMP3:%.*]] = load i32, i32* [[DOTOMP_LB]], align 4
// IR-NEXT:    store i32 [[TMP3]], i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND:.*]]
// IR:         [[OMP_INNER_FOR_COND]]:
// IR-NEXT:    [[TMP4:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[TMP5:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    [[CMP2:%.*]] = icmp sle i32 [[TMP4]], [[TMP5]]
// IR-NEXT:    br i1 [[CMP2]], label %[[OMP_INNER_FOR_BODY:.*]], label %[[OMP_INNER_FOR_END:.*]]
// IR:         [[OMP_INNER_FOR_BODY]]:
// IR-NEXT:    [[TMP6:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul nsw i32 [[TMP6]], 5
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 0, [[MUL]]
// IR-NEXT:    store i32 [[ADD]], i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 7, i32* [[I]], align 4
// IR-NEXT:    store i32 7, i32* [[J]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:         [[FOR_COND]]:
// IR-NEXT:    [[TMP7:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[CMP3:%.*]] = icmp slt i32 [[TMP7]], 4
// IR-NEXT:    br i1 [[CMP3]], label %[[FOR_BODY:.*]], label %[[FOR_END33:.*]]
// IR:         [[FOR_BODY]]:
// IR-NEXT:    [[TMP8:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP8]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4:.*]]
// IR:         [[FOR_COND4]]:
// IR-NEXT:    [[TMP9:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP10:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD5:%.*]] = add nsw i32 [[TMP10]], 5
// IR-NEXT:    [[CMP6:%.*]] = icmp slt i32 4, [[ADD5]]
// IR-NEXT:    br i1 [[CMP6]], label %[[COND_TRUE7:.*]], label %[[COND_FALSE8:.*]]
// IR:         [[COND_TRUE7]]:
// IR-NEXT:    br label %[[COND_END10:.*]]
// IR:         [[COND_FALSE8]]:
// IR-NEXT:    [[TMP11:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD9:%.*]] = add nsw i32 [[TMP11]], 5
// IR-NEXT:    br label %[[COND_END10]]
// IR:         [[COND_END10]]:
// IR-NEXT:    [[COND11:%.*]] = phi i32 [ 4, %[[COND_TRUE7]] ], [ [[ADD9]], %[[COND_FALSE8]] ]
// IR-NEXT:    [[CMP12:%.*]] = icmp slt i32 [[TMP9]], [[COND11]]
// IR-NEXT:    br i1 [[CMP12]], label %[[FOR_BODY13:.*]], label %[[FOR_END30:.*]]
// IR:         [[FOR_BODY13]]:
// IR-NEXT:    [[TMP12:%.+]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL13:%.+]] = mul nsw i32 [[TMP12]], 3
// IR-NEXT:    [[ADD14:%.+]] = add nsw i32 7, [[MUL13]]
// IR-NEXT:    store i32 [[ADD14]], i32* [[I]], align 4
// IR-NEXT:    [[TMP12:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    store i32 [[TMP12]], i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND14:.*]]
// IR:         [[FOR_COND14]]:
// IR-NEXT:    [[TMP13:%.*]] = load i32, i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[TMP14:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD15:%.*]] = add nsw i32 [[TMP14]], 5
// IR-NEXT:    [[CMP16:%.*]] = icmp slt i32 4, [[ADD15]]
// IR-NEXT:    br i1 [[CMP16]], label %[[COND_TRUE17:.*]], label %[[COND_FALSE18:.*]]
// IR:         [[COND_TRUE17]]:
// IR-NEXT:    br label %[[COND_END20:.*]]
// IR:         [[COND_FALSE18]]:
// IR-NEXT:    [[TMP15:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD19:%.*]] = add nsw i32 [[TMP15]], 5
// IR-NEXT:    br label %[[COND_END20]]
// IR:         [[COND_END20]]:
// IR-NEXT:    [[COND21:%.*]] = phi i32 [ 4, %[[COND_TRUE17]] ], [ [[ADD19]], %[[COND_FALSE18]] ]
// IR-NEXT:    [[CMP22:%.*]] = icmp slt i32 [[TMP13]], [[COND21]]
// IR-NEXT:    br i1 [[CMP22]], label %[[FOR_BODY23:.*]], label %[[FOR_END:.*]]
// IR:         [[FOR_BODY23]]:
// IR-NEXT:    [[TMP17:%.*]] = load i32, i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[MUL26:%.*]] = mul nsw i32 [[TMP17]], 3
// IR-NEXT:    [[ADD27:%.*]] = add nsw i32 7, [[MUL26]]
// IR-NEXT:    store i32 [[ADD27]], i32* [[J]], align 4
// IR-NEXT:    [[TMP18:%.*]] = load i32, i32* [[I]], align 4
// IR-NEXT:    [[TMP19:%.*]] = load i32, i32* [[J]], align 4
// IR-NEXT:    call void (...) @body(i32 [[TMP18]], i32 [[TMP19]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:         [[FOR_INC]]:
// IR-NEXT:    [[TMP20:%.*]] = load i32, i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP20]], 1
// IR-NEXT:    store i32 [[INC]], i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND14]]
// IR:         [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC28:.*]]
// IR:         [[FOR_INC28]]:
// IR-NEXT:    [[TMP21:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC29:%.*]] = add nsw i32 [[TMP21]], 1
// IR-NEXT:    store i32 [[INC29]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4]]
// IR:         [[FOR_END30]]:
// IR-NEXT:    br label %[[FOR_INC31:.*]]
// IR:         [[FOR_INC31]]:
// IR-NEXT:    [[TMP22:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD32:%.*]] = add nsw i32 [[TMP22]], 5
// IR-NEXT:    store i32 [[ADD32]], i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND]]
// IR:         [[FOR_END33]]:
// IR-NEXT:    br label %[[OMP_BODY_CONTINUE:.*]]
// IR:         [[OMP_BODY_CONTINUE]]:
// IR-NEXT:    br label %[[OMP_INNER_FOR_INC:.*]]
// IR:         [[OMP_INNER_FOR_INC]]:
// IR-NEXT:    [[TMP23:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[ADD34:%.*]] = add nsw i32 [[TMP23]], 1
// IR-NEXT:    store i32 [[ADD34]], i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// IR:         [[OMP_INNER_FOR_END]]:
// IR-NEXT:    br label %[[OMP_LOOP_EXIT:.*]]
// IR:         [[OMP_LOOP_EXIT]]:
// IR-NEXT:    call void @__kmpc_for_static_fini(%struct.ident_t* [[GLOB1]], i32 [[TMP0]])
// IR-NEXT:    call void @__kmpc_barrier(%struct.ident_t* [[GLOB3:@.*]], i32 [[TMP0]])
// IR-NEXT:    ret void
//
extern "C" void foo3() {
#pragma omp for
#pragma omp tile sizes(5,5)
    for (int i = 7; i < 17; i += 3)
      for (int j = 7; j < 17; j += 3)
        body(i, j);
}

// IR-LABEL: @foo4(
// IR-NEXT:  entry:
// IR-NEXT:    [[DOTOMP_IV:%.*]] = alloca i32, align 4
// IR-NEXT:    [[TMP:%.*]] = alloca i32, align 4
// IR-NEXT:    [[TMP1:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_LB:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_UB:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_STRIDE:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// IR-NEXT:    [[K:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_1_IV_J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_1_IV_J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[GLOB2]])
// IR-NEXT:    store i32 0, i32* [[DOTOMP_LB]], align 4
// IR-NEXT:    store i32 3, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    store i32 1, i32* [[DOTOMP_STRIDE]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTOMP_IS_LAST]], align 4
// IR-NEXT:    call void @__kmpc_for_static_init_4(%struct.ident_t* [[GLOB1]], i32 [[TMP0]], i32 34, i32* [[DOTOMP_IS_LAST]], i32* [[DOTOMP_LB]], i32* [[DOTOMP_UB]], i32* [[DOTOMP_STRIDE]], i32 1, i32 1)
// IR-NEXT:    [[TMP1:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[TMP1]], 3
// IR-NEXT:    br i1 [[CMP]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:         [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:         [[COND_FALSE]]:
// IR-NEXT:    [[TMP2:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    br label %[[COND_END]]
// IR:         [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 3, %[[COND_TRUE]] ], [ [[TMP2]], %[[COND_FALSE]] ]
// IR-NEXT:    store i32 [[COND]], i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    [[TMP3:%.*]] = load i32, i32* [[DOTOMP_LB]], align 4
// IR-NEXT:    store i32 [[TMP3]], i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND:.*]]
// IR:         [[OMP_INNER_FOR_COND]]:
// IR-NEXT:    [[TMP4:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[TMP5:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    [[CMP3:%.*]] = icmp sle i32 [[TMP4]], [[TMP5]]
// IR-NEXT:    br i1 [[CMP3]], label %[[OMP_INNER_FOR_BODY:.*]], label %[[OMP_INNER_FOR_END:.*]]
// IR:         [[OMP_INNER_FOR_BODY]]:
// IR-NEXT:    [[TMP6:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[DIV:%.*]] = sdiv i32 [[TMP6]], 1
// IR-NEXT:    [[MUL:%.*]] = mul nsw i32 [[DIV]], 3
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 7, [[MUL]]
// IR-NEXT:    store i32 [[ADD]], i32* [[K]], align 4
// IR-NEXT:    [[TMP7:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[TMP8:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[DIV4:%.*]] = sdiv i32 [[TMP8]], 1
// IR-NEXT:    [[MUL5:%.*]] = mul nsw i32 [[DIV4]], 1
// IR-NEXT:    [[SUB:%.*]] = sub nsw i32 [[TMP7]], [[MUL5]]
// IR-NEXT:    [[MUL6:%.*]] = mul nsw i32 [[SUB]], 5
// IR-NEXT:    [[ADD7:%.*]] = add nsw i32 0, [[MUL6]]
// IR-NEXT:    store i32 [[ADD7]], i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 7, i32* [[I]], align 4
// IR-NEXT:    store i32 7, i32* [[J]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:         [[FOR_COND]]:
// IR-NEXT:    [[TMP9:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[CMP8:%.*]] = icmp slt i32 [[TMP9]], 4
// IR-NEXT:    br i1 [[CMP8]], label %[[FOR_BODY:.*]], label %[[FOR_END38:.*]]
// IR:         [[FOR_BODY]]:
// IR-NEXT:    [[TMP10:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP10]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND9:.*]]
// IR:         [[FOR_COND9]]:
// IR-NEXT:    [[TMP11:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP12:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD10:%.*]] = add nsw i32 [[TMP12]], 5
// IR-NEXT:    [[CMP11:%.*]] = icmp slt i32 4, [[ADD10]]
// IR-NEXT:    br i1 [[CMP11]], label %[[COND_TRUE12:.*]], label %[[COND_FALSE13:.*]]
// IR:         [[COND_TRUE12]]:
// IR-NEXT:    br label %[[COND_END15:.*]]
// IR:         [[COND_FALSE13]]:
// IR-NEXT:    [[TMP13:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD14:%.*]] = add nsw i32 [[TMP13]], 5
// IR-NEXT:    br label %[[COND_END15]]
// IR:         [[COND_END15]]:
// IR-NEXT:    [[COND16:%.*]] = phi i32 [ 4, %[[COND_TRUE12]] ], [ [[ADD14]], %[[COND_FALSE13]] ]
// IR-NEXT:    [[CMP17:%.*]] = icmp slt i32 [[TMP11]], [[COND16]]
// IR-NEXT:    br i1 [[CMP17]], label %[[FOR_BODY18:.*]], label %[[FOR_END35:.*]]
// IR:         [[FOR_BODY18]]:
// IR-NEXT:    [[TMP14:%.+]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL18:%.+]] = mul nsw i32 [[TMP14]], 3
// IR-NEXT:    [[ADD19:%.+]] = add nsw i32 7, [[MUL18]]
// IR-NEXT:    store i32 [[ADD19]], i32* [[I]], align 4
// IR-NEXT:    [[TMP14:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    store i32 [[TMP14]], i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND19:.*]]
// IR:         [[FOR_COND19]]:
// IR-NEXT:    [[TMP15:%.*]] = load i32, i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[TMP16:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD20:%.*]] = add nsw i32 [[TMP16]], 5
// IR-NEXT:    [[CMP21:%.*]] = icmp slt i32 4, [[ADD20]]
// IR-NEXT:    br i1 [[CMP21]], label %[[COND_TRUE22:.*]], label %[[COND_FALSE23:.*]]
// IR:         [[COND_TRUE22]]:
// IR-NEXT:    br label %[[COND_END25:.*]]
// IR:         [[COND_FALSE23]]:
// IR-NEXT:    [[TMP17:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD24:%.*]] = add nsw i32 [[TMP17]], 5
// IR-NEXT:    br label %[[COND_END25]]
// IR:         [[COND_END25]]:
// IR-NEXT:    [[COND26:%.*]] = phi i32 [ 4, %[[COND_TRUE22]] ], [ [[ADD24]], %[[COND_FALSE23]] ]
// IR-NEXT:    [[CMP27:%.*]] = icmp slt i32 [[TMP15]], [[COND26]]
// IR-NEXT:    br i1 [[CMP27]], label %[[FOR_BODY28:.*]], label %[[FOR_END:.*]]
// IR:         [[FOR_BODY28]]:
// IR-NEXT:    [[TMP19:%.*]] = load i32, i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[MUL31:%.*]] = mul nsw i32 [[TMP19]], 3
// IR-NEXT:    [[ADD32:%.*]] = add nsw i32 7, [[MUL31]]
// IR-NEXT:    store i32 [[ADD32]], i32* [[J]], align 4
// IR-NEXT:    [[TMP20:%.*]] = load i32, i32* [[I]], align 4
// IR-NEXT:    [[TMP21:%.*]] = load i32, i32* [[J]], align 4
// IR-NEXT:    call void (...) @body(i32 [[TMP20]], i32 [[TMP21]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:         [[FOR_INC]]:
// IR-NEXT:    [[TMP22:%.*]] = load i32, i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP22]], 1
// IR-NEXT:    store i32 [[INC]], i32* [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND19]]
// IR:         [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC33:.*]]
// IR:         [[FOR_INC33]]:
// IR-NEXT:    [[TMP23:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC34:%.*]] = add nsw i32 [[TMP23]], 1
// IR-NEXT:    store i32 [[INC34]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND9]]
// IR:         [[FOR_END35]]:
// IR-NEXT:    br label %[[FOR_INC36:.*]]
// IR:         [[FOR_INC36]]:
// IR-NEXT:    [[TMP24:%.*]] = load i32, i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD37:%.*]] = add nsw i32 [[TMP24]], 5
// IR-NEXT:    store i32 [[ADD37]], i32* [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND]]
// IR:         [[FOR_END38]]:
// IR-NEXT:    br label %[[OMP_BODY_CONTINUE:.*]]
// IR:         [[OMP_BODY_CONTINUE]]:
// IR-NEXT:    br label %[[OMP_INNER_FOR_INC:.*]]
// IR:         [[OMP_INNER_FOR_INC]]:
// IR-NEXT:    [[TMP25:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[ADD39:%.*]] = add nsw i32 [[TMP25]], 1
// IR-NEXT:    store i32 [[ADD39]], i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// IR:         [[OMP_INNER_FOR_END]]:
// IR-NEXT:    br label %[[OMP_LOOP_EXIT:.*]]
// IR:         [[OMP_LOOP_EXIT]]:
// IR-NEXT:    call void @__kmpc_for_static_fini(%struct.ident_t* [[GLOB1]], i32 [[TMP0]])
// IR-NEXT:    call void @__kmpc_barrier(%struct.ident_t* [[GLOB3]], i32 [[TMP0]])
// IR-NEXT:    ret void
//
extern "C" void foo4() {
#pragma omp for collapse(2)
  for (int k = 7; k < 17; k += 3)
#pragma omp tile sizes(5,5)
  for (int i = 7; i < 17; i += 3)
    for (int j = 7; j < 17; j += 3)
      body(i, j);
}


// IR-LABEL: @foo5(
// IR-NEXT:  entry:
// IR-NEXT:    [[DOTOMP_IV:%.*]] = alloca i64, align 8
// IR-NEXT:    [[TMP:%.*]] = alloca i32, align 4
// IR-NEXT:    [[TMP1:%.*]] = alloca i32, align 4
// IR-NEXT:    [[TMP2:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTCAPTURE_EXPR_:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTCAPTURE_EXPR_3:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTCAPTURE_EXPR_5:%.*]] = alloca i64, align 8
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_LB:%.*]] = alloca i64, align 8
// IR-NEXT:    [[DOTOMP_UB:%.*]] = alloca i64, align 8
// IR-NEXT:    [[DOTOMP_STRIDE:%.*]] = alloca i64, align 8
// IR-NEXT:    [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I10:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I11:%.*]] = alloca i32, align 4
// IR-NEXT:    [[J15:%.*]] = alloca i32, align 4
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[GLOB2]])
// IR-NEXT:    [[TMP1:%.*]] = load i32, i32* [[TMP]], align 4
// IR-NEXT:    store i32 [[TMP1]], i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[TMP2:%.*]] = load i32, i32* [[TMP]], align 4
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP2]], 5
// IR-NEXT:    [[CMP:%.*]] = icmp slt i32 4, [[ADD]]
// IR-NEXT:    br i1 [[CMP]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:         [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:         [[COND_FALSE]]:
// IR-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP]], align 4
// IR-NEXT:    [[ADD4:%.*]] = add nsw i32 [[TMP3]], 5
// IR-NEXT:    br label %[[COND_END]]
// IR:         [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 4, %[[COND_TRUE]] ], [ [[ADD4]], %[[COND_FALSE]] ]
// IR-NEXT:    store i32 [[COND]], i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[TMP4:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[TMP5:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[SUB:%.*]] = sub i32 [[TMP4]], [[TMP5]]
// IR-NEXT:    [[SUB6:%.*]] = sub i32 [[SUB]], 1
// IR-NEXT:    [[ADD7:%.*]] = add i32 [[SUB6]], 1
// IR-NEXT:    [[DIV:%.*]] = udiv i32 [[ADD7]], 1
// IR-NEXT:    [[CONV:%.*]] = zext i32 [[DIV]] to i64
// IR-NEXT:    [[MUL:%.*]] = mul nsw i64 1, [[CONV]]
// IR-NEXT:    [[MUL8:%.*]] = mul nsw i64 [[MUL]], 4
// IR-NEXT:    [[SUB9:%.*]] = sub nsw i64 [[MUL8]], 1
// IR-NEXT:    store i64 [[SUB9]], i64* [[DOTCAPTURE_EXPR_5]], align 8
// IR-NEXT:    store i32 0, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[TMP6:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    store i32 [[TMP6]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    store i32 7, i32* [[J]], align 4
// IR-NEXT:    [[TMP7:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[TMP8:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[CMP12:%.*]] = icmp slt i32 [[TMP7]], [[TMP8]]
// IR-NEXT:    br i1 [[CMP12]], label %[[OMP_PRECOND_THEN:.*]], label %[[OMP_PRECOND_END:.*]]
// IR:         [[OMP_PRECOND_THEN]]:
// IR-NEXT:    store i64 0, i64* [[DOTOMP_LB]], align 8
// IR-NEXT:    [[TMP9:%.*]] = load i64, i64* [[DOTCAPTURE_EXPR_5]], align 8
// IR-NEXT:    store i64 [[TMP9]], i64* [[DOTOMP_UB]], align 8
// IR-NEXT:    store i64 1, i64* [[DOTOMP_STRIDE]], align 8
// IR-NEXT:    store i32 0, i32* [[DOTOMP_IS_LAST]], align 4
// IR-NEXT:    call void @__kmpc_for_static_init_8(%struct.ident_t* [[GLOB1]], i32 [[TMP0]], i32 34, i32* [[DOTOMP_IS_LAST]], i64* [[DOTOMP_LB]], i64* [[DOTOMP_UB]], i64* [[DOTOMP_STRIDE]], i64 1, i64 1)
// IR-NEXT:    [[TMP10:%.*]] = load i64, i64* [[DOTOMP_UB]], align 8
// IR-NEXT:    [[TMP11:%.*]] = load i64, i64* [[DOTCAPTURE_EXPR_5]], align 8
// IR-NEXT:    [[CMP16:%.*]] = icmp sgt i64 [[TMP10]], [[TMP11]]
// IR-NEXT:    br i1 [[CMP16]], label %[[COND_TRUE17:.*]], label %[[COND_FALSE18:.*]]
// IR:         [[COND_TRUE17]]:
// IR-NEXT:    [[TMP12:%.*]] = load i64, i64* [[DOTCAPTURE_EXPR_5]], align 8
// IR-NEXT:    br label %[[COND_END19:.*]]
// IR:         [[COND_FALSE18]]:
// IR-NEXT:    [[TMP13:%.*]] = load i64, i64* [[DOTOMP_UB]], align 8
// IR-NEXT:    br label %[[COND_END19]]
// IR:         [[COND_END19]]:
// IR-NEXT:    [[COND20:%.*]] = phi i64 [ [[TMP12]], %[[COND_TRUE17]] ], [ [[TMP13]], %[[COND_FALSE18]] ]
// IR-NEXT:    store i64 [[COND20]], i64* [[DOTOMP_UB]], align 8
// IR-NEXT:    [[TMP14:%.*]] = load i64, i64* [[DOTOMP_LB]], align 8
// IR-NEXT:    store i64 [[TMP14]], i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND:.*]]
// IR:         [[OMP_INNER_FOR_COND]]:
// IR-NEXT:    [[TMP15:%.*]] = load i64, i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    [[TMP16:%.*]] = load i64, i64* [[DOTOMP_UB]], align 8
// IR-NEXT:    [[CMP21:%.*]] = icmp sle i64 [[TMP15]], [[TMP16]]
// IR-NEXT:    br i1 [[CMP21]], label %[[OMP_INNER_FOR_BODY:.*]], label %[[OMP_INNER_FOR_END:.*]]
// IR:         [[OMP_INNER_FOR_BODY]]:
// IR-NEXT:    [[TMP17:%.*]] = load i64, i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    [[TMP18:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[TMP19:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[SUB22:%.*]] = sub i32 [[TMP18]], [[TMP19]]
// IR-NEXT:    [[SUB23:%.*]] = sub i32 [[SUB22]], 1
// IR-NEXT:    [[ADD24:%.*]] = add i32 [[SUB23]], 1
// IR-NEXT:    [[DIV25:%.*]] = udiv i32 [[ADD24]], 1
// IR-NEXT:    [[MUL26:%.*]] = mul i32 1, [[DIV25]]
// IR-NEXT:    [[MUL27:%.*]] = mul i32 [[MUL26]], 4
// IR-NEXT:    [[CONV28:%.*]] = zext i32 [[MUL27]] to i64
// IR-NEXT:    [[DIV29:%.*]] = sdiv i64 [[TMP17]], [[CONV28]]
// IR-NEXT:    [[MUL30:%.*]] = mul nsw i64 [[DIV29]], 5
// IR-NEXT:    [[ADD31:%.*]] = add nsw i64 0, [[MUL30]]
// IR-NEXT:    [[CONV32:%.*]] = trunc i64 [[ADD31]] to i32
// IR-NEXT:    store i32 [[CONV32]], i32* [[DOTFLOOR_0_IV_I10]], align 4
// IR-NEXT:    [[TMP20:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[CONV33:%.*]] = sext i32 [[TMP20]] to i64
// IR-NEXT:    [[TMP21:%.*]] = load i64, i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    [[TMP22:%.*]] = load i64, i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    [[TMP23:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[TMP24:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[SUB34:%.*]] = sub i32 [[TMP23]], [[TMP24]]
// IR-NEXT:    [[SUB35:%.*]] = sub i32 [[SUB34]], 1
// IR-NEXT:    [[ADD36:%.*]] = add i32 [[SUB35]], 1
// IR-NEXT:    [[DIV37:%.*]] = udiv i32 [[ADD36]], 1
// IR-NEXT:    [[MUL38:%.*]] = mul i32 1, [[DIV37]]
// IR-NEXT:    [[MUL39:%.*]] = mul i32 [[MUL38]], 4
// IR-NEXT:    [[CONV40:%.*]] = zext i32 [[MUL39]] to i64
// IR-NEXT:    [[DIV41:%.*]] = sdiv i64 [[TMP22]], [[CONV40]]
// IR-NEXT:    [[TMP25:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[TMP26:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[SUB42:%.*]] = sub i32 [[TMP25]], [[TMP26]]
// IR-NEXT:    [[SUB43:%.*]] = sub i32 [[SUB42]], 1
// IR-NEXT:    [[ADD44:%.*]] = add i32 [[SUB43]], 1
// IR-NEXT:    [[DIV45:%.*]] = udiv i32 [[ADD44]], 1
// IR-NEXT:    [[MUL46:%.*]] = mul i32 1, [[DIV45]]
// IR-NEXT:    [[MUL47:%.*]] = mul i32 [[MUL46]], 4
// IR-NEXT:    [[CONV48:%.*]] = zext i32 [[MUL47]] to i64
// IR-NEXT:    [[MUL49:%.*]] = mul nsw i64 [[DIV41]], [[CONV48]]
// IR-NEXT:    [[SUB50:%.*]] = sub nsw i64 [[TMP21]], [[MUL49]]
// IR-NEXT:    [[DIV51:%.*]] = sdiv i64 [[SUB50]], 4
// IR-NEXT:    [[MUL52:%.*]] = mul nsw i64 [[DIV51]], 1
// IR-NEXT:    [[ADD53:%.*]] = add nsw i64 [[CONV33]], [[MUL52]]
// IR-NEXT:    [[CONV54:%.*]] = trunc i64 [[ADD53]] to i32
// IR-NEXT:    store i32 [[CONV54]], i32* [[DOTTILE_0_IV_I11]], align 4
// IR-NEXT:    [[TMP27:%.*]] = load i64, i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    [[TMP28:%.*]] = load i64, i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    [[TMP29:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[TMP30:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[SUB55:%.*]] = sub i32 [[TMP29]], [[TMP30]]
// IR-NEXT:    [[SUB56:%.*]] = sub i32 [[SUB55]], 1
// IR-NEXT:    [[ADD57:%.*]] = add i32 [[SUB56]], 1
// IR-NEXT:    [[DIV58:%.*]] = udiv i32 [[ADD57]], 1
// IR-NEXT:    [[MUL59:%.*]] = mul i32 1, [[DIV58]]
// IR-NEXT:    [[MUL60:%.*]] = mul i32 [[MUL59]], 4
// IR-NEXT:    [[CONV61:%.*]] = zext i32 [[MUL60]] to i64
// IR-NEXT:    [[DIV62:%.*]] = sdiv i64 [[TMP28]], [[CONV61]]
// IR-NEXT:    [[TMP31:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[TMP32:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[SUB63:%.*]] = sub i32 [[TMP31]], [[TMP32]]
// IR-NEXT:    [[SUB64:%.*]] = sub i32 [[SUB63]], 1
// IR-NEXT:    [[ADD65:%.*]] = add i32 [[SUB64]], 1
// IR-NEXT:    [[DIV66:%.*]] = udiv i32 [[ADD65]], 1
// IR-NEXT:    [[MUL67:%.*]] = mul i32 1, [[DIV66]]
// IR-NEXT:    [[MUL68:%.*]] = mul i32 [[MUL67]], 4
// IR-NEXT:    [[CONV69:%.*]] = zext i32 [[MUL68]] to i64
// IR-NEXT:    [[MUL70:%.*]] = mul nsw i64 [[DIV62]], [[CONV69]]
// IR-NEXT:    [[SUB71:%.*]] = sub nsw i64 [[TMP27]], [[MUL70]]
// IR-NEXT:    [[TMP33:%.*]] = load i64, i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    [[TMP34:%.*]] = load i64, i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    [[TMP35:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[TMP36:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[SUB72:%.*]] = sub i32 [[TMP35]], [[TMP36]]
// IR-NEXT:    [[SUB73:%.*]] = sub i32 [[SUB72]], 1
// IR-NEXT:    [[ADD74:%.*]] = add i32 [[SUB73]], 1
// IR-NEXT:    [[DIV75:%.*]] = udiv i32 [[ADD74]], 1
// IR-NEXT:    [[MUL76:%.*]] = mul i32 1, [[DIV75]]
// IR-NEXT:    [[MUL77:%.*]] = mul i32 [[MUL76]], 4
// IR-NEXT:    [[CONV78:%.*]] = zext i32 [[MUL77]] to i64
// IR-NEXT:    [[DIV79:%.*]] = sdiv i64 [[TMP34]], [[CONV78]]
// IR-NEXT:    [[TMP37:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_3]], align 4
// IR-NEXT:    [[TMP38:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]], align 4
// IR-NEXT:    [[SUB80:%.*]] = sub i32 [[TMP37]], [[TMP38]]
// IR-NEXT:    [[SUB81:%.*]] = sub i32 [[SUB80]], 1
// IR-NEXT:    [[ADD82:%.*]] = add i32 [[SUB81]], 1
// IR-NEXT:    [[DIV83:%.*]] = udiv i32 [[ADD82]], 1
// IR-NEXT:    [[MUL84:%.*]] = mul i32 1, [[DIV83]]
// IR-NEXT:    [[MUL85:%.*]] = mul i32 [[MUL84]], 4
// IR-NEXT:    [[CONV86:%.*]] = zext i32 [[MUL85]] to i64
// IR-NEXT:    [[MUL87:%.*]] = mul nsw i64 [[DIV79]], [[CONV86]]
// IR-NEXT:    [[SUB88:%.*]] = sub nsw i64 [[TMP33]], [[MUL87]]
// IR-NEXT:    [[DIV89:%.*]] = sdiv i64 [[SUB88]], 4
// IR-NEXT:    [[MUL90:%.*]] = mul nsw i64 [[DIV89]], 4
// IR-NEXT:    [[SUB91:%.*]] = sub nsw i64 [[SUB71]], [[MUL90]]
// IR-NEXT:    [[MUL92:%.*]] = mul nsw i64 [[SUB91]], 3
// IR-NEXT:    [[ADD93:%.*]] = add nsw i64 7, [[MUL92]]
// IR-NEXT:    [[CONV94:%.*]] = trunc i64 [[ADD93]] to i32
// IR-NEXT:    store i32 [[CONV94]], i32* [[J15]], align 4
// IR-NEXT:    store i32 7, i32* [[I]], align 4
// IR-NEXT:    [[TMP39:%.*]] = load i32, i32* [[DOTTILE_0_IV_I11]], align 4
// IR-NEXT:    [[MUL95:%.*]] = mul nsw i32 [[TMP39]], 3
// IR-NEXT:    [[ADD96:%.*]] = add nsw i32 7, [[MUL95]]
// IR-NEXT:    store i32 [[ADD96]], i32* [[I]], align 4
// IR-NEXT:    [[TMP40:%.*]] = load i32, i32* [[I]], align 4
// IR-NEXT:    [[TMP41:%.*]] = load i32, i32* [[J15]], align 4
// IR-NEXT:    call void (...) @body(i32 [[TMP40]], i32 [[TMP41]])
// IR-NEXT:    br label %[[OMP_BODY_CONTINUE:.*]]
// IR:         [[OMP_BODY_CONTINUE]]:
// IR-NEXT:    br label %[[OMP_INNER_FOR_INC:.*]]
// IR:         [[OMP_INNER_FOR_INC]]:
// IR-NEXT:    [[TMP42:%.*]] = load i64, i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    [[ADD97:%.*]] = add nsw i64 [[TMP42]], 1
// IR-NEXT:    store i64 [[ADD97]], i64* [[DOTOMP_IV]], align 8
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// IR:         [[OMP_INNER_FOR_END]]:
// IR-NEXT:    br label %[[OMP_LOOP_EXIT:.*]]
// IR:         [[OMP_LOOP_EXIT]]:
// IR-NEXT:    call void @__kmpc_for_static_fini(%struct.ident_t* [[GLOB1]], i32 [[TMP0]])
// IR-NEXT:    br label %[[OMP_PRECOND_END]]
// IR:         [[OMP_PRECOND_END]]:
// IR-NEXT:    call void @__kmpc_barrier(%struct.ident_t* [[GLOB3]], i32 [[TMP0]])
// IR-NEXT:    ret void
//
extern "C" void foo5() {
#pragma omp for collapse(3)
#pragma omp tile sizes(5)
  for (int i = 7; i < 17; i += 3)
    for (int j = 7; j < 17; j += 3)
      body(i, j);
}


// IR-LABEL: @foo6(
// IR-NEXT:  entry:
// IR-NEXT:    call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[GLOB2]], i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @.omp_outlined. to void (i32*, i32*, ...)*))
// IR-NEXT:    ret void
//
// IR-LABEL: @.omp_outlined.(
// IR-NEXT:  entry:
// IR-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca i32*, align 8
// IR-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca i32*, align 8
// IR-NEXT:    [[DOTOMP_IV:%.*]] = alloca i32, align 4
// IR-NEXT:    [[TMP:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_LB:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_UB:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_STRIDE:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32* [[DOTGLOBAL_TID_:%.*]], i32** [[DOTGLOBAL_TID__ADDR]], align 8
// IR-NEXT:    store i32* [[DOTBOUND_TID_:%.*]], i32** [[DOTBOUND_TID__ADDR]], align 8
// IR-NEXT:    store i32 0, i32* [[DOTOMP_LB]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    store i32 1, i32* [[DOTOMP_STRIDE]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTOMP_IS_LAST]], align 4
// IR-NEXT:    [[TMP0:%.*]] = load i32*, i32** [[DOTGLOBAL_TID__ADDR]], align 8
// IR-NEXT:    [[TMP1:%.*]] = load i32, i32* [[TMP0]], align 4
// IR-NEXT:    call void @__kmpc_for_static_init_4(%struct.ident_t* [[GLOB1]], i32 [[TMP1]], i32 34, i32* [[DOTOMP_IS_LAST]], i32* [[DOTOMP_LB]], i32* [[DOTOMP_UB]], i32* [[DOTOMP_STRIDE]], i32 1, i32 1)
// IR-NEXT:    [[TMP2:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[TMP2]], 0
// IR-NEXT:    br i1 [[CMP]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:         [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:         [[COND_FALSE]]:
// IR-NEXT:    [[TMP3:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    br label %[[COND_END]]
// IR:         [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 0, %[[COND_TRUE]] ], [ [[TMP3]], %[[COND_FALSE]] ]
// IR-NEXT:    store i32 [[COND]], i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    [[TMP4:%.*]] = load i32, i32* [[DOTOMP_LB]], align 4
// IR-NEXT:    store i32 [[TMP4]], i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND:.*]]
// IR:         [[OMP_INNER_FOR_COND]]:
// IR-NEXT:    [[TMP5:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[TMP6:%.*]] = load i32, i32* [[DOTOMP_UB]], align 4
// IR-NEXT:    [[CMP2:%.*]] = icmp sle i32 [[TMP5]], [[TMP6]]
// IR-NEXT:    br i1 [[CMP2]], label %[[OMP_INNER_FOR_BODY:.*]], label %[[OMP_INNER_FOR_END:.*]]
// IR:         [[OMP_INNER_FOR_BODY]]:
// IR-NEXT:    [[TMP7:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul nsw i32 [[TMP7]], 5
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 0, [[MUL]]
// IR-NEXT:    store i32 [[ADD]], i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 7, i32* [[I]], align 4
// IR-NEXT:    [[TMP8:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP8]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:         [[FOR_COND]]:
// IR-NEXT:    [[TMP9:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP10:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD3:%.*]] = add nsw i32 [[TMP10]], 5
// IR-NEXT:    [[CMP4:%.*]] = icmp slt i32 4, [[ADD3]]
// IR-NEXT:    br i1 [[CMP4]], label %[[COND_TRUE5:.*]], label %[[COND_FALSE6:.*]]
// IR:         [[COND_TRUE5]]:
// IR-NEXT:    br label %[[COND_END8:.*]]
// IR:         [[COND_FALSE6]]:
// IR-NEXT:    [[TMP11:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD7:%.*]] = add nsw i32 [[TMP11]], 5
// IR-NEXT:    br label %[[COND_END8]]
// IR:         [[COND_END8]]:
// IR-NEXT:    [[COND9:%.*]] = phi i32 [ 4, %[[COND_TRUE5]] ], [ [[ADD7]], %[[COND_FALSE6]] ]
// IR-NEXT:    [[CMP10:%.*]] = icmp slt i32 [[TMP9]], [[COND9]]
// IR-NEXT:    br i1 [[CMP10]], label %[[FOR_BODY:.*]], label %[[FOR_END:.*]]
// IR:         [[FOR_BODY]]:
// IR-NEXT:    [[TMP12:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL11:%.*]] = mul nsw i32 [[TMP12]], 3
// IR-NEXT:    [[ADD12:%.*]] = add nsw i32 7, [[MUL11]]
// IR-NEXT:    store i32 [[ADD12]], i32* [[I]], align 4
// IR-NEXT:    [[TMP13:%.*]] = load i32, i32* [[I]], align 4
// IR-NEXT:    call void (...) @body(i32 [[TMP13]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:         [[FOR_INC]]:
// IR-NEXT:    [[TMP14:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP14]], 1
// IR-NEXT:    store i32 [[INC]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]]
// IR:         [[FOR_END]]:
// IR-NEXT:    br label %[[OMP_BODY_CONTINUE:.*]]
// IR:         [[OMP_BODY_CONTINUE]]:
// IR-NEXT:    br label %[[OMP_INNER_FOR_INC:.*]]
// IR:         [[OMP_INNER_FOR_INC]]:
// IR-NEXT:    [[TMP15:%.*]] = load i32, i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    [[ADD13:%.*]] = add nsw i32 [[TMP15]], 1
// IR-NEXT:    store i32 [[ADD13]], i32* [[DOTOMP_IV]], align 4
// IR-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// IR:         [[OMP_INNER_FOR_END]]:
// IR-NEXT:    br label %[[OMP_LOOP_EXIT:.*]]
// IR:         [[OMP_LOOP_EXIT]]:
// IR-NEXT:    call void @__kmpc_for_static_fini(%struct.ident_t* [[GLOB1]], i32 [[TMP1]])
// IR-NEXT:    ret void
//
extern "C" void foo6() {
#pragma omp parallel for
#pragma omp tile sizes(5)
  for (int i = 7; i < 17; i += 3)
    body(i);
}


template<typename T, T Step, T Tile>
void foo7(T start, T end) {
#pragma omp tile sizes(Tile)
  for (T i = start; i < end; i += Step)
    body(i);
}

// IR-LABEL: define {{.*}}void @tfoo7(
// IR-NEXT:  entry:
// IR-NEXT:    call void @_Z4foo7IiLi3ELi5EEvT_S0_(i32 0, i32 42)
// IR-NEXT:    ret void
//
// IR-LABEL: define linkonce_odr void @_Z4foo7IiLi3ELi5EEvT_S0_(
// IR-NEXT:  entry:
// IR-NEXT:    [[START_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[END_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[CAPTURE_EXPR:%.+]] = alloca i32, align 4
// IR-NEXT:    [[CAPTURE_EXPR1:%.+]] = alloca i32, align 4
// IR-NEXT:    [[CAPTURE_EXPR2:%.+]] = alloca i32, align 4
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32 [[START:%.*]], i32* [[START_ADDR]], align 4
// IR-NEXT:    store i32 [[END:%.*]], i32* [[END_ADDR]], align 4
// IR-NEXT:    [[TMP0:%.+]] = load i32, i32* [[START_ADDR]], align 4
// IR-NEXT:    store i32 [[TMP0]], i32* [[CAPTURE_EXPR]], align 4
// IR-NEXT:    [[TMP1:%.+]] = load i32, i32* [[END_ADDR]], align 4
// IR-NEXT:    store i32 [[TMP1]], i32* [[CAPTURE_EXPR1]], align 4
// IR-NEXT:    [[TMP2:%.+]] = load i32, i32* [[CAPTURE_EXPR1]], align 4
// IR-NEXT:    [[TMP3:%.+]] = load i32, i32* [[CAPTURE_EXPR]], align 4
// IR-NEXT:    [[SUB:%.+]] = sub i32 [[TMP2]], [[TMP3]]
// IR-NEXT:    [[SUB3:%.+]] = sub i32 [[SUB]], 1
// IR-NEXT:    [[ADD:%.+]] = add i32 [[SUB3]], 3
// IR-NEXT:    [[DIV:%.+]] = udiv i32 [[ADD]], 3
// IR-NEXT:    [[SUB4:%.+]] = sub i32 [[DIV]], 1
// IR-NEXT:    store i32 [[SUB4]], i32* [[CAPTURE_EXPR2]], align 4
// IR-NEXT:    [[TMP4:%.+]] = load i32, i32* [[START_ADDR]], align 4
// IR-NEXT:    store i32 [[TMP4]], i32* [[I]], align 4
// IR-NEXT:    store i32 0, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:         [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[TMP6:%.+]] = load i32, i32* [[CAPTURE_EXPR2]], align 4
// IR-NEXT:    [[ADD3:%.*]] = add i32 [[TMP6]], 1
// IR-NEXT:    [[CMP:%.*]] = icmp ult i32 [[TMP0]], [[ADD3]]
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END25:.*]]
// IR:         [[FOR_BODY]]:
// IR-NEXT:    [[TMP3:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP3]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4:.*]]
// IR:         [[FOR_COND4]]:
// IR-NEXT:    [[TMP4:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP5:%.*]] = load i32, i32* [[CAPTURE_EXPR2]], align 4
// IR-NEXT:    [[ADD10:%.*]] = add i32 [[TMP5]], 1
// IR-NEXT:    [[TMP7:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD11:%.*]] = add nsw i32 [[TMP7]], 5
// IR-NEXT:    [[CMP12:%.*]] = icmp ult i32 [[ADD10]], [[ADD11]]
// IR-NEXT:    br i1 [[CMP12]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:         [[COND_TRUE]]:
// IR-NEXT:    [[TMP8:%.*]] = load i32, i32* [[CAPTURE_EXPR2]], align 4
// IR-NEXT:    [[ADD18:%.*]] = add i32 [[TMP8]], 1
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:         [[COND_FALSE]]:
// IR-NEXT:    [[TMP10:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD19:%.*]] = add nsw i32 [[TMP10]], 5
// IR-NEXT:    br label %[[COND_END]]
// IR:         [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ [[ADD18]], %[[COND_TRUE]] ], [ [[ADD19]], %[[COND_FALSE]] ]
// IR-NEXT:    [[CMP20:%.*]] = icmp ult i32 [[TMP4]], [[COND]]
// IR-NEXT:    br i1 [[CMP20]], label %[[FOR_BODY21:.*]], label %[[FOR_END:.*]]
// IR:         [[FOR_BODY21]]:
// IR-NEXT:    [[TMP11:%.*]] = load i32, i32* [[CAPTURE_EXPR]], align 4
// IR-NEXT:    [[TMP13:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul i32 [[TMP13]], 3
// IR-NEXT:    [[ADD22:%.*]] = add i32 [[TMP11]], [[MUL]]
// IR-NEXT:    store i32 [[ADD22]], i32* [[I]], align 4
// IR-NEXT:    [[TMP14:%.*]] = load i32, i32* [[I]], align 4
// IR-NEXT:    call void (...) @body(i32 [[TMP14]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:         [[FOR_INC]]:
// IR-NEXT:    [[TMP15:%.*]] = load i32, i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP15]], 1
// IR-NEXT:    store i32 [[INC]], i32* [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4]]
// IR:         [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC23:.*]]
// IR:         [[FOR_INC23]]:
// IR-NEXT:    [[TMP16:%.*]] = load i32, i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD24:%.*]] = add nsw i32 [[TMP16]], 5
// IR-NEXT:    store i32 [[ADD24]], i32* [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]]
// IR:         [[FOR_END25]]:
// IR-NEXT:    ret void
//
extern "C" void tfoo7() {
  foo7<int,3,5>(0, 42);
}

#endif /* HEADER */
