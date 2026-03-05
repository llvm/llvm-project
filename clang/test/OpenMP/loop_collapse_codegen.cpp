// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown \
// RUN: -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

// CHECK-LABEL: define internal void @_Z17triangulat_loop_1v.omp_outlined(
// CHECK-SAME: ptr noalias noundef [[DOTGLOBAL_TID_:%.*]], ptr noalias noundef [[DOTBOUND_TID_:%.*]]) #[[ATTR1:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTOMP_IV:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[TMP:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[_TMP1:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[_TMP2:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MIN:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MAX:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTMIN_LESS_MAX:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[DOTLOWER:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MIN4:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MAX7:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTMIN_LESS_MAX13:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[DOTLOWER16:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTCAPTURE_EXPR_:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_LB:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_UB:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_STRIDE:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[J:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[K:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[DOTBOUND_TID__ADDR]], align 8
// CHECK-NEXT:    store i32 0, ptr [[TMP]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD:%.*]] = add i32 [[TMP0]], 1
// CHECK-NEXT:    store i32 [[ADD]], ptr [[DOTLB_MIN]], align 4
// CHECK-NEXT:    store i32 9, ptr [[TMP]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD3:%.*]] = add i32 [[TMP1]], 1
// CHECK-NEXT:    store i32 [[ADD3]], ptr [[DOTLB_MAX]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTLB_MIN]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTLB_MAX]], align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp ult i32 [[TMP2]], [[TMP3]]
// CHECK-NEXT:    [[STOREDV:%.*]] = zext i1 [[CMP]] to i8
// CHECK-NEXT:    store i8 [[STOREDV]], ptr [[DOTMIN_LESS_MAX]], align 1
// CHECK-NEXT:    [[TMP4:%.*]] = load i8, ptr [[DOTMIN_LESS_MAX]], align 1
// CHECK-NEXT:    [[LOADEDV:%.*]] = trunc i8 [[TMP4]] to i1
// CHECK-NEXT:    br i1 [[LOADEDV]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// CHECK:       [[COND_TRUE]]:
// CHECK-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTLB_MIN]], align 4
// CHECK-NEXT:    br label %[[COND_END:.*]]
// CHECK:       [[COND_FALSE]]:
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, ptr [[DOTLB_MAX]], align 4
// CHECK-NEXT:    br label %[[COND_END]]
// CHECK:       [[COND_END]]:
// CHECK-NEXT:    [[COND:%.*]] = phi i32 [ [[TMP5]], %[[COND_TRUE]] ], [ [[TMP6]], %[[COND_FALSE]] ]
// CHECK-NEXT:    store i32 [[COND]], ptr [[TMP]], align 4
// CHECK-NEXT:    store i32 [[COND]], ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD5:%.*]] = add i32 [[TMP7]], 1
// CHECK-NEXT:    store i32 [[ADD5]], ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[ADD6:%.*]] = add i32 [[TMP8]], 1
// CHECK-NEXT:    store i32 [[ADD6]], ptr [[DOTLB_MIN4]], align 4
// CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD8:%.*]] = add i32 [[TMP9]], 1
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD9:%.*]] = add i32 [[TMP10]], 1
// CHECK-NEXT:    [[SUB:%.*]] = sub i32 10, [[ADD9]]
// CHECK-NEXT:    [[ADD10:%.*]] = add i32 [[SUB]], 1
// CHECK-NEXT:    [[MUL:%.*]] = mul i32 [[ADD10]], 1
// CHECK-NEXT:    [[ADD11:%.*]] = add i32 [[ADD8]], [[MUL]]
// CHECK-NEXT:    store i32 [[ADD11]], ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[ADD12:%.*]] = add i32 [[TMP11]], 1
// CHECK-NEXT:    store i32 [[ADD12]], ptr [[DOTLB_MAX7]], align 4
// CHECK-NEXT:    [[TMP12:%.*]] = load i32, ptr [[DOTLB_MIN4]], align 4
// CHECK-NEXT:    [[TMP13:%.*]] = load i32, ptr [[DOTLB_MAX7]], align 4
// CHECK-NEXT:    [[CMP14:%.*]] = icmp ult i32 [[TMP12]], [[TMP13]]
// CHECK-NEXT:    [[STOREDV15:%.*]] = zext i1 [[CMP14]] to i8
// CHECK-NEXT:    store i8 [[STOREDV15]], ptr [[DOTMIN_LESS_MAX13]], align 1
// CHECK-NEXT:    [[TMP14:%.*]] = load i8, ptr [[DOTMIN_LESS_MAX13]], align 1
// CHECK-NEXT:    [[LOADEDV17:%.*]] = trunc i8 [[TMP14]] to i1
// CHECK-NEXT:    br i1 [[LOADEDV17]], label %[[COND_TRUE18:.*]], label %[[COND_FALSE19:.*]]
// CHECK:       [[COND_TRUE18]]:
// CHECK-NEXT:    [[TMP15:%.*]] = load i32, ptr [[DOTLB_MIN4]], align 4
// CHECK-NEXT:    br label %[[COND_END20:.*]]
// CHECK:       [[COND_FALSE19]]:
// CHECK-NEXT:    [[TMP16:%.*]] = load i32, ptr [[DOTLB_MAX7]], align 4
// CHECK-NEXT:    br label %[[COND_END20]]
// CHECK:       [[COND_END20]]:
// CHECK-NEXT:    [[COND21:%.*]] = phi i32 [ [[TMP15]], %[[COND_TRUE18]] ], [ [[TMP16]], %[[COND_FALSE19]] ]
// CHECK-NEXT:    store i32 [[COND21]], ptr [[_TMP1]], align 4
// CHECK-NEXT:    store i32 [[COND21]], ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[TMP17:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB22:%.*]] = sub i32 10, [[TMP17]]
// CHECK-NEXT:    [[ADD23:%.*]] = add i32 [[SUB22]], 1
// CHECK-NEXT:    [[CONV:%.*]] = zext i32 [[ADD23]] to i64
// CHECK-NEXT:    [[MUL24:%.*]] = mul nsw i64 10, [[CONV]]
// CHECK-NEXT:    [[TMP18:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB25:%.*]] = sub i32 10, [[TMP18]]
// CHECK-NEXT:    [[ADD26:%.*]] = add i32 [[SUB25]], 1
// CHECK-NEXT:    [[CONV27:%.*]] = zext i32 [[ADD26]] to i64
// CHECK-NEXT:    [[MUL28:%.*]] = mul nsw i64 [[MUL24]], [[CONV27]]
// CHECK-NEXT:    [[SUB29:%.*]] = sub nsw i64 [[MUL28]], 1
// CHECK-NEXT:    store i64 [[SUB29]], ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    store i64 0, ptr [[DOTOMP_LB]], align 8
// CHECK-NEXT:    [[TMP19:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    store i64 [[TMP19]], ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    store i64 1, ptr [[DOTOMP_STRIDE]], align 8
// CHECK-NEXT:    store i32 0, ptr [[DOTOMP_IS_LAST]], align 4
// CHECK-NEXT:    [[TMP20:%.*]] = load ptr, ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    [[TMP21:%.*]] = load i32, ptr [[TMP20]], align 4
// CHECK-NEXT:    call void @__kmpc_for_static_init_8(ptr @[[GLOB1:[0-9]+]], i32 [[TMP21]], i32 34, ptr [[DOTOMP_IS_LAST]], ptr [[DOTOMP_LB]], ptr [[DOTOMP_UB]], ptr [[DOTOMP_STRIDE]], i64 1, i64 1)
// CHECK-NEXT:    [[TMP22:%.*]] = load i64, ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    [[TMP23:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    [[CMP30:%.*]] = icmp sgt i64 [[TMP22]], [[TMP23]]
// CHECK-NEXT:    br i1 [[CMP30]], label %[[COND_TRUE31:.*]], label %[[COND_FALSE32:.*]]
// CHECK:       [[COND_TRUE31]]:
// CHECK-NEXT:    [[TMP24:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    br label %[[COND_END33:.*]]
// CHECK:       [[COND_FALSE32]]:
// CHECK-NEXT:    [[TMP25:%.*]] = load i64, ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    br label %[[COND_END33]]
// CHECK:       [[COND_END33]]:
// CHECK-NEXT:    [[COND34:%.*]] = phi i64 [ [[TMP24]], %[[COND_TRUE31]] ], [ [[TMP25]], %[[COND_FALSE32]] ]
// CHECK-NEXT:    store i64 [[COND34]], ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    [[TMP26:%.*]] = load i64, ptr [[DOTOMP_LB]], align 8
// CHECK-NEXT:    store i64 [[TMP26]], ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_COND:.*]]
// CHECK:       [[OMP_INNER_FOR_COND]]:
// CHECK-NEXT:    [[TMP27:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP28:%.*]] = load i64, ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    [[CMP35:%.*]] = icmp sle i64 [[TMP27]], [[TMP28]]
// CHECK-NEXT:    br i1 [[CMP35]], label %[[OMP_INNER_FOR_BODY:.*]], label %[[OMP_INNER_FOR_END:.*]]
// CHECK:       [[OMP_INNER_FOR_BODY]]:
// CHECK-NEXT:    [[TMP29:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP30:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB36:%.*]] = sub i32 10, [[TMP30]]
// CHECK-NEXT:    [[ADD37:%.*]] = add i32 [[SUB36]], 1
// CHECK-NEXT:    [[MUL38:%.*]] = mul i32 1, [[ADD37]]
// CHECK-NEXT:    [[TMP31:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB39:%.*]] = sub i32 10, [[TMP31]]
// CHECK-NEXT:    [[ADD40:%.*]] = add i32 [[SUB39]], 1
// CHECK-NEXT:    [[MUL41:%.*]] = mul i32 [[MUL38]], [[ADD40]]
// CHECK-NEXT:    [[CONV42:%.*]] = zext i32 [[MUL41]] to i64
// CHECK-NEXT:    [[DIV:%.*]] = sdiv i64 [[TMP29]], [[CONV42]]
// CHECK-NEXT:    [[MUL43:%.*]] = mul nsw i64 [[DIV]], 1
// CHECK-NEXT:    [[ADD44:%.*]] = add nsw i64 0, [[MUL43]]
// CHECK-NEXT:    [[CONV45:%.*]] = trunc i64 [[ADD44]] to i32
// CHECK-NEXT:    store i32 [[CONV45]], ptr [[I]], align 4
// CHECK-NEXT:    [[TMP32:%.*]] = load i32, ptr [[I]], align 4
// CHECK-NEXT:    [[ADD46:%.*]] = add i32 [[TMP32]], 1
// CHECK-NEXT:    [[CONV47:%.*]] = zext i32 [[ADD46]] to i64
// CHECK-NEXT:    [[TMP33:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP34:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP35:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB48:%.*]] = sub i32 10, [[TMP35]]
// CHECK-NEXT:    [[ADD49:%.*]] = add i32 [[SUB48]], 1
// CHECK-NEXT:    [[MUL50:%.*]] = mul i32 1, [[ADD49]]
// CHECK-NEXT:    [[TMP36:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB51:%.*]] = sub i32 10, [[TMP36]]
// CHECK-NEXT:    [[ADD52:%.*]] = add i32 [[SUB51]], 1
// CHECK-NEXT:    [[MUL53:%.*]] = mul i32 [[MUL50]], [[ADD52]]
// CHECK-NEXT:    [[CONV54:%.*]] = zext i32 [[MUL53]] to i64
// CHECK-NEXT:    [[DIV55:%.*]] = sdiv i64 [[TMP34]], [[CONV54]]
// CHECK-NEXT:    [[TMP37:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB56:%.*]] = sub i32 10, [[TMP37]]
// CHECK-NEXT:    [[ADD57:%.*]] = add i32 [[SUB56]], 1
// CHECK-NEXT:    [[MUL58:%.*]] = mul i32 1, [[ADD57]]
// CHECK-NEXT:    [[TMP38:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB59:%.*]] = sub i32 10, [[TMP38]]
// CHECK-NEXT:    [[ADD60:%.*]] = add i32 [[SUB59]], 1
// CHECK-NEXT:    [[MUL61:%.*]] = mul i32 [[MUL58]], [[ADD60]]
// CHECK-NEXT:    [[CONV62:%.*]] = zext i32 [[MUL61]] to i64
// CHECK-NEXT:    [[MUL63:%.*]] = mul nsw i64 [[DIV55]], [[CONV62]]
// CHECK-NEXT:    [[SUB64:%.*]] = sub nsw i64 [[TMP33]], [[MUL63]]
// CHECK-NEXT:    [[TMP39:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB65:%.*]] = sub i32 10, [[TMP39]]
// CHECK-NEXT:    [[ADD66:%.*]] = add i32 [[SUB65]], 1
// CHECK-NEXT:    [[MUL67:%.*]] = mul i32 1, [[ADD66]]
// CHECK-NEXT:    [[CONV68:%.*]] = zext i32 [[MUL67]] to i64
// CHECK-NEXT:    [[DIV69:%.*]] = sdiv i64 [[SUB64]], [[CONV68]]
// CHECK-NEXT:    [[MUL70:%.*]] = mul nsw i64 [[DIV69]], 1
// CHECK-NEXT:    [[ADD71:%.*]] = add nsw i64 [[CONV47]], [[MUL70]]
// CHECK-NEXT:    [[CONV72:%.*]] = trunc i64 [[ADD71]] to i32
// CHECK-NEXT:    store i32 [[CONV72]], ptr [[J]], align 4
// CHECK-NEXT:    [[TMP40:%.*]] = load i32, ptr [[J]], align 4
// CHECK-NEXT:    [[ADD73:%.*]] = add i32 [[TMP40]], 1
// CHECK-NEXT:    [[CONV74:%.*]] = zext i32 [[ADD73]] to i64
// CHECK-NEXT:    [[TMP41:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP42:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP43:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB75:%.*]] = sub i32 10, [[TMP43]]
// CHECK-NEXT:    [[ADD76:%.*]] = add i32 [[SUB75]], 1
// CHECK-NEXT:    [[MUL77:%.*]] = mul i32 1, [[ADD76]]
// CHECK-NEXT:    [[TMP44:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB78:%.*]] = sub i32 10, [[TMP44]]
// CHECK-NEXT:    [[ADD79:%.*]] = add i32 [[SUB78]], 1
// CHECK-NEXT:    [[MUL80:%.*]] = mul i32 [[MUL77]], [[ADD79]]
// CHECK-NEXT:    [[CONV81:%.*]] = zext i32 [[MUL80]] to i64
// CHECK-NEXT:    [[DIV82:%.*]] = sdiv i64 [[TMP42]], [[CONV81]]
// CHECK-NEXT:    [[TMP45:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB83:%.*]] = sub i32 10, [[TMP45]]
// CHECK-NEXT:    [[ADD84:%.*]] = add i32 [[SUB83]], 1
// CHECK-NEXT:    [[MUL85:%.*]] = mul i32 1, [[ADD84]]
// CHECK-NEXT:    [[TMP46:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB86:%.*]] = sub i32 10, [[TMP46]]
// CHECK-NEXT:    [[ADD87:%.*]] = add i32 [[SUB86]], 1
// CHECK-NEXT:    [[MUL88:%.*]] = mul i32 [[MUL85]], [[ADD87]]
// CHECK-NEXT:    [[CONV89:%.*]] = zext i32 [[MUL88]] to i64
// CHECK-NEXT:    [[MUL90:%.*]] = mul nsw i64 [[DIV82]], [[CONV89]]
// CHECK-NEXT:    [[SUB91:%.*]] = sub nsw i64 [[TMP41]], [[MUL90]]
// CHECK-NEXT:    [[TMP47:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP48:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP49:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB92:%.*]] = sub i32 10, [[TMP49]]
// CHECK-NEXT:    [[ADD93:%.*]] = add i32 [[SUB92]], 1
// CHECK-NEXT:    [[MUL94:%.*]] = mul i32 1, [[ADD93]]
// CHECK-NEXT:    [[TMP50:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB95:%.*]] = sub i32 10, [[TMP50]]
// CHECK-NEXT:    [[ADD96:%.*]] = add i32 [[SUB95]], 1
// CHECK-NEXT:    [[MUL97:%.*]] = mul i32 [[MUL94]], [[ADD96]]
// CHECK-NEXT:    [[CONV98:%.*]] = zext i32 [[MUL97]] to i64
// CHECK-NEXT:    [[DIV99:%.*]] = sdiv i64 [[TMP48]], [[CONV98]]
// CHECK-NEXT:    [[TMP51:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB100:%.*]] = sub i32 10, [[TMP51]]
// CHECK-NEXT:    [[ADD101:%.*]] = add i32 [[SUB100]], 1
// CHECK-NEXT:    [[MUL102:%.*]] = mul i32 1, [[ADD101]]
// CHECK-NEXT:    [[TMP52:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB103:%.*]] = sub i32 10, [[TMP52]]
// CHECK-NEXT:    [[ADD104:%.*]] = add i32 [[SUB103]], 1
// CHECK-NEXT:    [[MUL105:%.*]] = mul i32 [[MUL102]], [[ADD104]]
// CHECK-NEXT:    [[CONV106:%.*]] = zext i32 [[MUL105]] to i64
// CHECK-NEXT:    [[MUL107:%.*]] = mul nsw i64 [[DIV99]], [[CONV106]]
// CHECK-NEXT:    [[SUB108:%.*]] = sub nsw i64 [[TMP47]], [[MUL107]]
// CHECK-NEXT:    [[TMP53:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB109:%.*]] = sub i32 10, [[TMP53]]
// CHECK-NEXT:    [[ADD110:%.*]] = add i32 [[SUB109]], 1
// CHECK-NEXT:    [[MUL111:%.*]] = mul i32 1, [[ADD110]]
// CHECK-NEXT:    [[CONV112:%.*]] = zext i32 [[MUL111]] to i64
// CHECK-NEXT:    [[DIV113:%.*]] = sdiv i64 [[SUB108]], [[CONV112]]
// CHECK-NEXT:    [[TMP54:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB114:%.*]] = sub i32 10, [[TMP54]]
// CHECK-NEXT:    [[ADD115:%.*]] = add i32 [[SUB114]], 1
// CHECK-NEXT:    [[MUL116:%.*]] = mul i32 1, [[ADD115]]
// CHECK-NEXT:    [[CONV117:%.*]] = zext i32 [[MUL116]] to i64
// CHECK-NEXT:    [[MUL118:%.*]] = mul nsw i64 [[DIV113]], [[CONV117]]
// CHECK-NEXT:    [[SUB119:%.*]] = sub nsw i64 [[SUB91]], [[MUL118]]
// CHECK-NEXT:    [[MUL120:%.*]] = mul nsw i64 [[SUB119]], 1
// CHECK-NEXT:    [[ADD121:%.*]] = add nsw i64 [[CONV74]], [[MUL120]]
// CHECK-NEXT:    [[CONV122:%.*]] = trunc i64 [[ADD121]] to i32
// CHECK-NEXT:    store i32 [[CONV122]], ptr [[K]], align 4
// CHECK-NEXT:    [[TMP55:%.*]] = load i32, ptr [[J]], align 4
// CHECK-NEXT:    [[CMP123:%.*]] = icmp ult i32 [[TMP55]], 10
// CHECK-NEXT:    br i1 [[CMP123]], label %[[OMP_BODY_NEXT:.*]], label %[[OMP_BODY_CONTINUE:.*]]
// CHECK:       [[OMP_BODY_NEXT]]:
// CHECK-NEXT:    [[TMP56:%.*]] = load i32, ptr [[K]], align 4
// CHECK-NEXT:    [[CMP124:%.*]] = icmp ult i32 [[TMP56]], 10
// CHECK-NEXT:    br i1 [[CMP124]], label %[[OMP_BODY_NEXT125:.*]], label %[[OMP_BODY_CONTINUE]]
// CHECK:       [[OMP_BODY_NEXT125]]:
// CHECK-NEXT:    br label %[[OMP_BODY_CONTINUE]]
// CHECK:       [[OMP_BODY_CONTINUE]]:
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_INC:.*]]
// CHECK:       [[OMP_INNER_FOR_INC]]:
// CHECK-NEXT:    [[TMP57:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[ADD126:%.*]] = add nsw i64 [[TMP57]], 1
// CHECK-NEXT:    store i64 [[ADD126]], ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// CHECK:       [[OMP_INNER_FOR_END]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_EXIT:.*]]
// CHECK:       [[OMP_LOOP_EXIT]]:
// CHECK-NEXT:    [[TMP58:%.*]] = load ptr, ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    [[TMP59:%.*]] = load i32, ptr [[TMP58]], align 4
// CHECK-NEXT:    call void @__kmpc_for_static_fini(ptr @[[GLOB1]], i32 [[TMP59]])
// CHECK-NEXT:    ret void
void triangulat_loop_1() {
#pragma omp parallel for collapse(3)
  for (unsigned int i = 0; i < 10; ++i)
    for (unsigned int j = i + 1; j < 10; ++j)
      for (unsigned int k = j + 1; k < 10; ++k)
	;
}

// CHECK-LABEL: define internal void @_Z17triangulat_loop_2v.omp_outlined(
// CHECK-SAME: ptr noalias noundef [[DOTGLOBAL_TID_:%.*]], ptr noalias noundef [[DOTBOUND_TID_:%.*]]) #[[ATTR1]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTOMP_IV:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[TMP:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[_TMP1:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[_TMP2:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MIN:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MAX:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTMIN_LESS_MAX:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[DOTLOWER:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MIN4:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MAX7:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTMIN_LESS_MAX13:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[DOTLOWER16:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTCAPTURE_EXPR_:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_LB:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_UB:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_STRIDE:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[J:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[K:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[DOTBOUND_TID__ADDR]], align 8
// CHECK-NEXT:    store i32 0, ptr [[TMP]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD:%.*]] = add i32 [[TMP0]], 1
// CHECK-NEXT:    store i32 [[ADD]], ptr [[DOTLB_MIN]], align 4
// CHECK-NEXT:    store i32 9, ptr [[TMP]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD3:%.*]] = add i32 [[TMP1]], 1
// CHECK-NEXT:    store i32 [[ADD3]], ptr [[DOTLB_MAX]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTLB_MIN]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTLB_MAX]], align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP2]], [[TMP3]]
// CHECK-NEXT:    [[STOREDV:%.*]] = zext i1 [[CMP]] to i8
// CHECK-NEXT:    store i8 [[STOREDV]], ptr [[DOTMIN_LESS_MAX]], align 1
// CHECK-NEXT:    [[TMP4:%.*]] = load i8, ptr [[DOTMIN_LESS_MAX]], align 1
// CHECK-NEXT:    [[LOADEDV:%.*]] = trunc i8 [[TMP4]] to i1
// CHECK-NEXT:    br i1 [[LOADEDV]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// CHECK:       [[COND_TRUE]]:
// CHECK-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTLB_MIN]], align 4
// CHECK-NEXT:    br label %[[COND_END:.*]]
// CHECK:       [[COND_FALSE]]:
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, ptr [[DOTLB_MAX]], align 4
// CHECK-NEXT:    br label %[[COND_END]]
// CHECK:       [[COND_END]]:
// CHECK-NEXT:    [[COND:%.*]] = phi i32 [ [[TMP5]], %[[COND_TRUE]] ], [ [[TMP6]], %[[COND_FALSE]] ]
// CHECK-NEXT:    store i32 [[COND]], ptr [[TMP]], align 4
// CHECK-NEXT:    store i32 [[COND]], ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD5:%.*]] = add i32 [[TMP7]], 1
// CHECK-NEXT:    store i32 [[ADD5]], ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[ADD6:%.*]] = add nsw i32 [[TMP8]], 1
// CHECK-NEXT:    store i32 [[ADD6]], ptr [[DOTLB_MIN4]], align 4
// CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD8:%.*]] = add i32 [[TMP9]], 1
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, ptr [[TMP]], align 4
// CHECK-NEXT:    [[ADD9:%.*]] = add i32 [[TMP10]], 1
// CHECK-NEXT:    [[SUB:%.*]] = sub i32 10, [[ADD9]]
// CHECK-NEXT:    [[ADD10:%.*]] = add i32 [[SUB]], 1
// CHECK-NEXT:    [[MUL:%.*]] = mul i32 [[ADD10]], 2
// CHECK-NEXT:    [[ADD11:%.*]] = add i32 [[ADD8]], [[MUL]]
// CHECK-NEXT:    store i32 [[ADD11]], ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[ADD12:%.*]] = add nsw i32 [[TMP11]], 1
// CHECK-NEXT:    store i32 [[ADD12]], ptr [[DOTLB_MAX7]], align 4
// CHECK-NEXT:    [[TMP12:%.*]] = load i32, ptr [[DOTLB_MIN4]], align 4
// CHECK-NEXT:    [[TMP13:%.*]] = load i32, ptr [[DOTLB_MAX7]], align 4
// CHECK-NEXT:    [[CMP14:%.*]] = icmp ult i32 [[TMP12]], [[TMP13]]
// CHECK-NEXT:    [[STOREDV15:%.*]] = zext i1 [[CMP14]] to i8
// CHECK-NEXT:    store i8 [[STOREDV15]], ptr [[DOTMIN_LESS_MAX13]], align 1
// CHECK-NEXT:    [[TMP14:%.*]] = load i8, ptr [[DOTMIN_LESS_MAX13]], align 1
// CHECK-NEXT:    [[LOADEDV17:%.*]] = trunc i8 [[TMP14]] to i1
// CHECK-NEXT:    br i1 [[LOADEDV17]], label %[[COND_TRUE18:.*]], label %[[COND_FALSE19:.*]]
// CHECK:       [[COND_TRUE18]]:
// CHECK-NEXT:    [[TMP15:%.*]] = load i32, ptr [[DOTLB_MIN4]], align 4
// CHECK-NEXT:    br label %[[COND_END20:.*]]
// CHECK:       [[COND_FALSE19]]:
// CHECK-NEXT:    [[TMP16:%.*]] = load i32, ptr [[DOTLB_MAX7]], align 4
// CHECK-NEXT:    br label %[[COND_END20]]
// CHECK:       [[COND_END20]]:
// CHECK-NEXT:    [[COND21:%.*]] = phi i32 [ [[TMP15]], %[[COND_TRUE18]] ], [ [[TMP16]], %[[COND_FALSE19]] ]
// CHECK-NEXT:    store i32 [[COND21]], ptr [[_TMP1]], align 4
// CHECK-NEXT:    store i32 [[COND21]], ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[TMP17:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB22:%.*]] = sub i32 10, [[TMP17]]
// CHECK-NEXT:    [[ADD23:%.*]] = add i32 [[SUB22]], 1
// CHECK-NEXT:    [[CONV:%.*]] = zext i32 [[ADD23]] to i64
// CHECK-NEXT:    [[MUL24:%.*]] = mul nsw i64 10, [[CONV]]
// CHECK-NEXT:    [[TMP18:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB25:%.*]] = sub i32 10, [[TMP18]]
// CHECK-NEXT:    [[ADD26:%.*]] = add i32 [[SUB25]], 1
// CHECK-NEXT:    [[CONV27:%.*]] = zext i32 [[ADD26]] to i64
// CHECK-NEXT:    [[MUL28:%.*]] = mul nsw i64 [[MUL24]], [[CONV27]]
// CHECK-NEXT:    [[SUB29:%.*]] = sub nsw i64 [[MUL28]], 1
// CHECK-NEXT:    store i64 [[SUB29]], ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    store i64 0, ptr [[DOTOMP_LB]], align 8
// CHECK-NEXT:    [[TMP19:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    store i64 [[TMP19]], ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    store i64 1, ptr [[DOTOMP_STRIDE]], align 8
// CHECK-NEXT:    store i32 0, ptr [[DOTOMP_IS_LAST]], align 4
// CHECK-NEXT:    [[TMP20:%.*]] = load ptr, ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    [[TMP21:%.*]] = load i32, ptr [[TMP20]], align 4
// CHECK-NEXT:    call void @__kmpc_for_static_init_8(ptr @[[GLOB1]], i32 [[TMP21]], i32 34, ptr [[DOTOMP_IS_LAST]], ptr [[DOTOMP_LB]], ptr [[DOTOMP_UB]], ptr [[DOTOMP_STRIDE]], i64 1, i64 1)
// CHECK-NEXT:    [[TMP22:%.*]] = load i64, ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    [[TMP23:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    [[CMP30:%.*]] = icmp sgt i64 [[TMP22]], [[TMP23]]
// CHECK-NEXT:    br i1 [[CMP30]], label %[[COND_TRUE31:.*]], label %[[COND_FALSE32:.*]]
// CHECK:       [[COND_TRUE31]]:
// CHECK-NEXT:    [[TMP24:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    br label %[[COND_END33:.*]]
// CHECK:       [[COND_FALSE32]]:
// CHECK-NEXT:    [[TMP25:%.*]] = load i64, ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    br label %[[COND_END33]]
// CHECK:       [[COND_END33]]:
// CHECK-NEXT:    [[COND34:%.*]] = phi i64 [ [[TMP24]], %[[COND_TRUE31]] ], [ [[TMP25]], %[[COND_FALSE32]] ]
// CHECK-NEXT:    store i64 [[COND34]], ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    [[TMP26:%.*]] = load i64, ptr [[DOTOMP_LB]], align 8
// CHECK-NEXT:    store i64 [[TMP26]], ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_COND:.*]]
// CHECK:       [[OMP_INNER_FOR_COND]]:
// CHECK-NEXT:    [[TMP27:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP28:%.*]] = load i64, ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    [[CMP35:%.*]] = icmp sle i64 [[TMP27]], [[TMP28]]
// CHECK-NEXT:    br i1 [[CMP35]], label %[[OMP_INNER_FOR_BODY:.*]], label %[[OMP_INNER_FOR_END:.*]]
// CHECK:       [[OMP_INNER_FOR_BODY]]:
// CHECK-NEXT:    [[TMP29:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP30:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB36:%.*]] = sub i32 10, [[TMP30]]
// CHECK-NEXT:    [[ADD37:%.*]] = add i32 [[SUB36]], 1
// CHECK-NEXT:    [[MUL38:%.*]] = mul i32 1, [[ADD37]]
// CHECK-NEXT:    [[TMP31:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB39:%.*]] = sub i32 10, [[TMP31]]
// CHECK-NEXT:    [[ADD40:%.*]] = add i32 [[SUB39]], 1
// CHECK-NEXT:    [[MUL41:%.*]] = mul i32 [[MUL38]], [[ADD40]]
// CHECK-NEXT:    [[CONV42:%.*]] = zext i32 [[MUL41]] to i64
// CHECK-NEXT:    [[DIV:%.*]] = sdiv i64 [[TMP29]], [[CONV42]]
// CHECK-NEXT:    [[MUL43:%.*]] = mul nsw i64 [[DIV]], 1
// CHECK-NEXT:    [[ADD44:%.*]] = add nsw i64 0, [[MUL43]]
// CHECK-NEXT:    [[CONV45:%.*]] = trunc i64 [[ADD44]] to i32
// CHECK-NEXT:    store i32 [[CONV45]], ptr [[I]], align 4
// CHECK-NEXT:    [[TMP32:%.*]] = load i32, ptr [[I]], align 4
// CHECK-NEXT:    [[ADD46:%.*]] = add i32 [[TMP32]], 1
// CHECK-NEXT:    [[CONV47:%.*]] = sext i32 [[ADD46]] to i64
// CHECK-NEXT:    [[TMP33:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP34:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP35:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB48:%.*]] = sub i32 10, [[TMP35]]
// CHECK-NEXT:    [[ADD49:%.*]] = add i32 [[SUB48]], 1
// CHECK-NEXT:    [[MUL50:%.*]] = mul i32 1, [[ADD49]]
// CHECK-NEXT:    [[TMP36:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB51:%.*]] = sub i32 10, [[TMP36]]
// CHECK-NEXT:    [[ADD52:%.*]] = add i32 [[SUB51]], 1
// CHECK-NEXT:    [[MUL53:%.*]] = mul i32 [[MUL50]], [[ADD52]]
// CHECK-NEXT:    [[CONV54:%.*]] = zext i32 [[MUL53]] to i64
// CHECK-NEXT:    [[DIV55:%.*]] = sdiv i64 [[TMP34]], [[CONV54]]
// CHECK-NEXT:    [[TMP37:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB56:%.*]] = sub i32 10, [[TMP37]]
// CHECK-NEXT:    [[ADD57:%.*]] = add i32 [[SUB56]], 1
// CHECK-NEXT:    [[MUL58:%.*]] = mul i32 1, [[ADD57]]
// CHECK-NEXT:    [[TMP38:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB59:%.*]] = sub i32 10, [[TMP38]]
// CHECK-NEXT:    [[ADD60:%.*]] = add i32 [[SUB59]], 1
// CHECK-NEXT:    [[MUL61:%.*]] = mul i32 [[MUL58]], [[ADD60]]
// CHECK-NEXT:    [[CONV62:%.*]] = zext i32 [[MUL61]] to i64
// CHECK-NEXT:    [[MUL63:%.*]] = mul nsw i64 [[DIV55]], [[CONV62]]
// CHECK-NEXT:    [[SUB64:%.*]] = sub nsw i64 [[TMP33]], [[MUL63]]
// CHECK-NEXT:    [[TMP39:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB65:%.*]] = sub i32 10, [[TMP39]]
// CHECK-NEXT:    [[ADD66:%.*]] = add i32 [[SUB65]], 1
// CHECK-NEXT:    [[MUL67:%.*]] = mul i32 1, [[ADD66]]
// CHECK-NEXT:    [[CONV68:%.*]] = zext i32 [[MUL67]] to i64
// CHECK-NEXT:    [[DIV69:%.*]] = sdiv i64 [[SUB64]], [[CONV68]]
// CHECK-NEXT:    [[MUL70:%.*]] = mul nsw i64 [[DIV69]], 2
// CHECK-NEXT:    [[ADD71:%.*]] = add nsw i64 [[CONV47]], [[MUL70]]
// CHECK-NEXT:    [[CONV72:%.*]] = trunc i64 [[ADD71]] to i32
// CHECK-NEXT:    store i32 [[CONV72]], ptr [[J]], align 4
// CHECK-NEXT:    [[TMP40:%.*]] = load i32, ptr [[J]], align 4
// CHECK-NEXT:    [[ADD73:%.*]] = add nsw i32 [[TMP40]], 1
// CHECK-NEXT:    [[CONV74:%.*]] = zext i32 [[ADD73]] to i64
// CHECK-NEXT:    [[TMP41:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP42:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP43:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB75:%.*]] = sub i32 10, [[TMP43]]
// CHECK-NEXT:    [[ADD76:%.*]] = add i32 [[SUB75]], 1
// CHECK-NEXT:    [[MUL77:%.*]] = mul i32 1, [[ADD76]]
// CHECK-NEXT:    [[TMP44:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB78:%.*]] = sub i32 10, [[TMP44]]
// CHECK-NEXT:    [[ADD79:%.*]] = add i32 [[SUB78]], 1
// CHECK-NEXT:    [[MUL80:%.*]] = mul i32 [[MUL77]], [[ADD79]]
// CHECK-NEXT:    [[CONV81:%.*]] = zext i32 [[MUL80]] to i64
// CHECK-NEXT:    [[DIV82:%.*]] = sdiv i64 [[TMP42]], [[CONV81]]
// CHECK-NEXT:    [[TMP45:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB83:%.*]] = sub i32 10, [[TMP45]]
// CHECK-NEXT:    [[ADD84:%.*]] = add i32 [[SUB83]], 1
// CHECK-NEXT:    [[MUL85:%.*]] = mul i32 1, [[ADD84]]
// CHECK-NEXT:    [[TMP46:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB86:%.*]] = sub i32 10, [[TMP46]]
// CHECK-NEXT:    [[ADD87:%.*]] = add i32 [[SUB86]], 1
// CHECK-NEXT:    [[MUL88:%.*]] = mul i32 [[MUL85]], [[ADD87]]
// CHECK-NEXT:    [[CONV89:%.*]] = zext i32 [[MUL88]] to i64
// CHECK-NEXT:    [[MUL90:%.*]] = mul nsw i64 [[DIV82]], [[CONV89]]
// CHECK-NEXT:    [[SUB91:%.*]] = sub nsw i64 [[TMP41]], [[MUL90]]
// CHECK-NEXT:    [[TMP47:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP48:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP49:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB92:%.*]] = sub i32 10, [[TMP49]]
// CHECK-NEXT:    [[ADD93:%.*]] = add i32 [[SUB92]], 1
// CHECK-NEXT:    [[MUL94:%.*]] = mul i32 1, [[ADD93]]
// CHECK-NEXT:    [[TMP50:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB95:%.*]] = sub i32 10, [[TMP50]]
// CHECK-NEXT:    [[ADD96:%.*]] = add i32 [[SUB95]], 1
// CHECK-NEXT:    [[MUL97:%.*]] = mul i32 [[MUL94]], [[ADD96]]
// CHECK-NEXT:    [[CONV98:%.*]] = zext i32 [[MUL97]] to i64
// CHECK-NEXT:    [[DIV99:%.*]] = sdiv i64 [[TMP48]], [[CONV98]]
// CHECK-NEXT:    [[TMP51:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB100:%.*]] = sub i32 10, [[TMP51]]
// CHECK-NEXT:    [[ADD101:%.*]] = add i32 [[SUB100]], 1
// CHECK-NEXT:    [[MUL102:%.*]] = mul i32 1, [[ADD101]]
// CHECK-NEXT:    [[TMP52:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB103:%.*]] = sub i32 10, [[TMP52]]
// CHECK-NEXT:    [[ADD104:%.*]] = add i32 [[SUB103]], 1
// CHECK-NEXT:    [[MUL105:%.*]] = mul i32 [[MUL102]], [[ADD104]]
// CHECK-NEXT:    [[CONV106:%.*]] = zext i32 [[MUL105]] to i64
// CHECK-NEXT:    [[MUL107:%.*]] = mul nsw i64 [[DIV99]], [[CONV106]]
// CHECK-NEXT:    [[SUB108:%.*]] = sub nsw i64 [[TMP47]], [[MUL107]]
// CHECK-NEXT:    [[TMP53:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB109:%.*]] = sub i32 10, [[TMP53]]
// CHECK-NEXT:    [[ADD110:%.*]] = add i32 [[SUB109]], 1
// CHECK-NEXT:    [[MUL111:%.*]] = mul i32 1, [[ADD110]]
// CHECK-NEXT:    [[CONV112:%.*]] = zext i32 [[MUL111]] to i64
// CHECK-NEXT:    [[DIV113:%.*]] = sdiv i64 [[SUB108]], [[CONV112]]
// CHECK-NEXT:    [[TMP54:%.*]] = load i32, ptr [[DOTLOWER16]], align 4
// CHECK-NEXT:    [[SUB114:%.*]] = sub i32 10, [[TMP54]]
// CHECK-NEXT:    [[ADD115:%.*]] = add i32 [[SUB114]], 1
// CHECK-NEXT:    [[MUL116:%.*]] = mul i32 1, [[ADD115]]
// CHECK-NEXT:    [[CONV117:%.*]] = zext i32 [[MUL116]] to i64
// CHECK-NEXT:    [[MUL118:%.*]] = mul nsw i64 [[DIV113]], [[CONV117]]
// CHECK-NEXT:    [[SUB119:%.*]] = sub nsw i64 [[SUB91]], [[MUL118]]
// CHECK-NEXT:    [[MUL120:%.*]] = mul nsw i64 [[SUB119]], 1
// CHECK-NEXT:    [[ADD121:%.*]] = add nsw i64 [[CONV74]], [[MUL120]]
// CHECK-NEXT:    [[CONV122:%.*]] = trunc i64 [[ADD121]] to i32
// CHECK-NEXT:    store i32 [[CONV122]], ptr [[K]], align 4
// CHECK-NEXT:    [[TMP55:%.*]] = load i32, ptr [[J]], align 4
// CHECK-NEXT:    [[CMP123:%.*]] = icmp slt i32 [[TMP55]], 10
// CHECK-NEXT:    br i1 [[CMP123]], label %[[OMP_BODY_NEXT:.*]], label %[[OMP_BODY_CONTINUE:.*]]
// CHECK:       [[OMP_BODY_NEXT]]:
// CHECK-NEXT:    [[TMP56:%.*]] = load i32, ptr [[K]], align 4
// CHECK-NEXT:    [[CMP124:%.*]] = icmp ult i32 [[TMP56]], 10
// CHECK-NEXT:    br i1 [[CMP124]], label %[[OMP_BODY_NEXT125:.*]], label %[[OMP_BODY_CONTINUE]]
// CHECK:       [[OMP_BODY_NEXT125]]:
// CHECK-NEXT:    br label %[[OMP_BODY_CONTINUE]]
// CHECK:       [[OMP_BODY_CONTINUE]]:
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_INC:.*]]
// CHECK:       [[OMP_INNER_FOR_INC]]:
// CHECK-NEXT:    [[TMP57:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[ADD126:%.*]] = add nsw i64 [[TMP57]], 1
// CHECK-NEXT:    store i64 [[ADD126]], ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// CHECK:       [[OMP_INNER_FOR_END]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_EXIT:.*]]
// CHECK:       [[OMP_LOOP_EXIT]]:
// CHECK-NEXT:    [[TMP58:%.*]] = load ptr, ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    [[TMP59:%.*]] = load i32, ptr [[TMP58]], align 4
// CHECK-NEXT:    call void @__kmpc_for_static_fini(ptr @[[GLOB1]], i32 [[TMP59]])
// CHECK-NEXT:    ret void
void triangulat_loop_2() {
#pragma omp parallel for collapse(3)
  for (unsigned int i = 0; i < 10; ++i)
    for (int j = i + 1; j < 10; j += 2)
      for (unsigned int k = j + 1; k < 10; ++k)
	;
}

// CHECK-LABEL: define internal void @_Z10mixed_loopv.omp_outlined(
// CHECK-SAME: ptr noalias noundef [[DOTGLOBAL_TID_:%.*]], ptr noalias noundef [[DOTBOUND_TID_:%.*]]) #[[ATTR1]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTOMP_IV:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[TMP:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[_TMP1:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[_TMP2:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MIN:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTLB_MAX:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTMIN_LESS_MAX:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[DOTLOWER:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTCAPTURE_EXPR_:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_LB:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_UB:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_STRIDE:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[J:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[K:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[DOTBOUND_TID__ADDR]], align 8
// CHECK-NEXT:    store i32 0, ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP0]], 1
// CHECK-NEXT:    store i32 [[ADD]], ptr [[DOTLB_MIN]], align 4
// CHECK-NEXT:    store i32 9, ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[_TMP1]], align 4
// CHECK-NEXT:    [[ADD3:%.*]] = add nsw i32 [[TMP1]], 1
// CHECK-NEXT:    store i32 [[ADD3]], ptr [[DOTLB_MAX]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTLB_MIN]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTLB_MAX]], align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP2]], [[TMP3]]
// CHECK-NEXT:    [[STOREDV:%.*]] = zext i1 [[CMP]] to i8
// CHECK-NEXT:    store i8 [[STOREDV]], ptr [[DOTMIN_LESS_MAX]], align 1
// CHECK-NEXT:    [[TMP4:%.*]] = load i8, ptr [[DOTMIN_LESS_MAX]], align 1
// CHECK-NEXT:    [[LOADEDV:%.*]] = trunc i8 [[TMP4]] to i1
// CHECK-NEXT:    br i1 [[LOADEDV]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// CHECK:       [[COND_TRUE]]:
// CHECK-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTLB_MIN]], align 4
// CHECK-NEXT:    br label %[[COND_END:.*]]
// CHECK:       [[COND_FALSE]]:
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, ptr [[DOTLB_MAX]], align 4
// CHECK-NEXT:    br label %[[COND_END]]
// CHECK:       [[COND_END]]:
// CHECK-NEXT:    [[COND:%.*]] = phi i32 [ [[TMP5]], %[[COND_TRUE]] ], [ [[TMP6]], %[[COND_FALSE]] ]
// CHECK-NEXT:    store i32 [[COND]], ptr [[_TMP1]], align 4
// CHECK-NEXT:    store i32 [[COND]], ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB:%.*]] = sub i32 10, [[TMP7]]
// CHECK-NEXT:    [[ADD4:%.*]] = add i32 [[SUB]], 1
// CHECK-NEXT:    [[CONV:%.*]] = zext i32 [[ADD4]] to i64
// CHECK-NEXT:    [[MUL:%.*]] = mul nsw i64 100, [[CONV]]
// CHECK-NEXT:    [[SUB5:%.*]] = sub nsw i64 [[MUL]], 1
// CHECK-NEXT:    store i64 [[SUB5]], ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    store i64 0, ptr [[DOTOMP_LB]], align 8
// CHECK-NEXT:    [[TMP8:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    store i64 [[TMP8]], ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    store i64 1, ptr [[DOTOMP_STRIDE]], align 8
// CHECK-NEXT:    store i32 0, ptr [[DOTOMP_IS_LAST]], align 4
// CHECK-NEXT:    [[TMP9:%.*]] = load ptr, ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, ptr [[TMP9]], align 4
// CHECK-NEXT:    call void @__kmpc_for_static_init_8(ptr @[[GLOB1]], i32 [[TMP10]], i32 34, ptr [[DOTOMP_IS_LAST]], ptr [[DOTOMP_LB]], ptr [[DOTOMP_UB]], ptr [[DOTOMP_STRIDE]], i64 1, i64 1)
// CHECK-NEXT:    [[TMP11:%.*]] = load i64, ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    [[TMP12:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    [[CMP6:%.*]] = icmp sgt i64 [[TMP11]], [[TMP12]]
// CHECK-NEXT:    br i1 [[CMP6]], label %[[COND_TRUE7:.*]], label %[[COND_FALSE8:.*]]
// CHECK:       [[COND_TRUE7]]:
// CHECK-NEXT:    [[TMP13:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR_]], align 8
// CHECK-NEXT:    br label %[[COND_END9:.*]]
// CHECK:       [[COND_FALSE8]]:
// CHECK-NEXT:    [[TMP14:%.*]] = load i64, ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    br label %[[COND_END9]]
// CHECK:       [[COND_END9]]:
// CHECK-NEXT:    [[COND10:%.*]] = phi i64 [ [[TMP13]], %[[COND_TRUE7]] ], [ [[TMP14]], %[[COND_FALSE8]] ]
// CHECK-NEXT:    store i64 [[COND10]], ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    [[TMP15:%.*]] = load i64, ptr [[DOTOMP_LB]], align 8
// CHECK-NEXT:    store i64 [[TMP15]], ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_COND:.*]]
// CHECK:       [[OMP_INNER_FOR_COND]]:
// CHECK-NEXT:    [[TMP16:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP17:%.*]] = load i64, ptr [[DOTOMP_UB]], align 8
// CHECK-NEXT:    [[CMP11:%.*]] = icmp sle i64 [[TMP16]], [[TMP17]]
// CHECK-NEXT:    br i1 [[CMP11]], label %[[OMP_INNER_FOR_BODY:.*]], label %[[OMP_INNER_FOR_END:.*]]
// CHECK:       [[OMP_INNER_FOR_BODY]]:
// CHECK-NEXT:    [[TMP18:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP19:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB12:%.*]] = sub i32 10, [[TMP19]]
// CHECK-NEXT:    [[ADD13:%.*]] = add i32 [[SUB12]], 1
// CHECK-NEXT:    [[MUL14:%.*]] = mul i32 10, [[ADD13]]
// CHECK-NEXT:    [[CONV15:%.*]] = zext i32 [[MUL14]] to i64
// CHECK-NEXT:    [[DIV:%.*]] = sdiv i64 [[TMP18]], [[CONV15]]
// CHECK-NEXT:    [[MUL16:%.*]] = mul nsw i64 [[DIV]], 1
// CHECK-NEXT:    [[ADD17:%.*]] = add nsw i64 0, [[MUL16]]
// CHECK-NEXT:    [[CONV18:%.*]] = trunc i64 [[ADD17]] to i32
// CHECK-NEXT:    store i32 [[CONV18]], ptr [[I]], align 4
// CHECK-NEXT:    [[TMP20:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP21:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP22:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB19:%.*]] = sub i32 10, [[TMP22]]
// CHECK-NEXT:    [[ADD20:%.*]] = add i32 [[SUB19]], 1
// CHECK-NEXT:    [[MUL21:%.*]] = mul i32 10, [[ADD20]]
// CHECK-NEXT:    [[CONV22:%.*]] = zext i32 [[MUL21]] to i64
// CHECK-NEXT:    [[DIV23:%.*]] = sdiv i64 [[TMP21]], [[CONV22]]
// CHECK-NEXT:    [[TMP23:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB24:%.*]] = sub i32 10, [[TMP23]]
// CHECK-NEXT:    [[ADD25:%.*]] = add i32 [[SUB24]], 1
// CHECK-NEXT:    [[MUL26:%.*]] = mul i32 10, [[ADD25]]
// CHECK-NEXT:    [[CONV27:%.*]] = zext i32 [[MUL26]] to i64
// CHECK-NEXT:    [[MUL28:%.*]] = mul nsw i64 [[DIV23]], [[CONV27]]
// CHECK-NEXT:    [[SUB29:%.*]] = sub nsw i64 [[TMP20]], [[MUL28]]
// CHECK-NEXT:    [[TMP24:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB30:%.*]] = sub i32 10, [[TMP24]]
// CHECK-NEXT:    [[ADD31:%.*]] = add i32 [[SUB30]], 1
// CHECK-NEXT:    [[MUL32:%.*]] = mul i32 1, [[ADD31]]
// CHECK-NEXT:    [[CONV33:%.*]] = zext i32 [[MUL32]] to i64
// CHECK-NEXT:    [[DIV34:%.*]] = sdiv i64 [[SUB29]], [[CONV33]]
// CHECK-NEXT:    [[MUL35:%.*]] = mul nsw i64 [[DIV34]], 1
// CHECK-NEXT:    [[ADD36:%.*]] = add nsw i64 0, [[MUL35]]
// CHECK-NEXT:    [[CONV37:%.*]] = trunc i64 [[ADD36]] to i32
// CHECK-NEXT:    store i32 [[CONV37]], ptr [[J]], align 4
// CHECK-NEXT:    [[TMP25:%.*]] = load i32, ptr [[J]], align 4
// CHECK-NEXT:    [[ADD38:%.*]] = add nsw i32 [[TMP25]], 1
// CHECK-NEXT:    [[CONV39:%.*]] = sext i32 [[ADD38]] to i64
// CHECK-NEXT:    [[TMP26:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP27:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP28:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB40:%.*]] = sub i32 10, [[TMP28]]
// CHECK-NEXT:    [[ADD41:%.*]] = add i32 [[SUB40]], 1
// CHECK-NEXT:    [[MUL42:%.*]] = mul i32 10, [[ADD41]]
// CHECK-NEXT:    [[CONV43:%.*]] = zext i32 [[MUL42]] to i64
// CHECK-NEXT:    [[DIV44:%.*]] = sdiv i64 [[TMP27]], [[CONV43]]
// CHECK-NEXT:    [[TMP29:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB45:%.*]] = sub i32 10, [[TMP29]]
// CHECK-NEXT:    [[ADD46:%.*]] = add i32 [[SUB45]], 1
// CHECK-NEXT:    [[MUL47:%.*]] = mul i32 10, [[ADD46]]
// CHECK-NEXT:    [[CONV48:%.*]] = zext i32 [[MUL47]] to i64
// CHECK-NEXT:    [[MUL49:%.*]] = mul nsw i64 [[DIV44]], [[CONV48]]
// CHECK-NEXT:    [[SUB50:%.*]] = sub nsw i64 [[TMP26]], [[MUL49]]
// CHECK-NEXT:    [[TMP30:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP31:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[TMP32:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB51:%.*]] = sub i32 10, [[TMP32]]
// CHECK-NEXT:    [[ADD52:%.*]] = add i32 [[SUB51]], 1
// CHECK-NEXT:    [[MUL53:%.*]] = mul i32 10, [[ADD52]]
// CHECK-NEXT:    [[CONV54:%.*]] = zext i32 [[MUL53]] to i64
// CHECK-NEXT:    [[DIV55:%.*]] = sdiv i64 [[TMP31]], [[CONV54]]
// CHECK-NEXT:    [[TMP33:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB56:%.*]] = sub i32 10, [[TMP33]]
// CHECK-NEXT:    [[ADD57:%.*]] = add i32 [[SUB56]], 1
// CHECK-NEXT:    [[MUL58:%.*]] = mul i32 10, [[ADD57]]
// CHECK-NEXT:    [[CONV59:%.*]] = zext i32 [[MUL58]] to i64
// CHECK-NEXT:    [[MUL60:%.*]] = mul nsw i64 [[DIV55]], [[CONV59]]
// CHECK-NEXT:    [[SUB61:%.*]] = sub nsw i64 [[TMP30]], [[MUL60]]
// CHECK-NEXT:    [[TMP34:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB62:%.*]] = sub i32 10, [[TMP34]]
// CHECK-NEXT:    [[ADD63:%.*]] = add i32 [[SUB62]], 1
// CHECK-NEXT:    [[MUL64:%.*]] = mul i32 1, [[ADD63]]
// CHECK-NEXT:    [[CONV65:%.*]] = zext i32 [[MUL64]] to i64
// CHECK-NEXT:    [[DIV66:%.*]] = sdiv i64 [[SUB61]], [[CONV65]]
// CHECK-NEXT:    [[TMP35:%.*]] = load i32, ptr [[DOTLOWER]], align 4
// CHECK-NEXT:    [[SUB67:%.*]] = sub i32 10, [[TMP35]]
// CHECK-NEXT:    [[ADD68:%.*]] = add i32 [[SUB67]], 1
// CHECK-NEXT:    [[MUL69:%.*]] = mul i32 1, [[ADD68]]
// CHECK-NEXT:    [[CONV70:%.*]] = zext i32 [[MUL69]] to i64
// CHECK-NEXT:    [[MUL71:%.*]] = mul nsw i64 [[DIV66]], [[CONV70]]
// CHECK-NEXT:    [[SUB72:%.*]] = sub nsw i64 [[SUB50]], [[MUL71]]
// CHECK-NEXT:    [[MUL73:%.*]] = mul nsw i64 [[SUB72]], 1
// CHECK-NEXT:    [[ADD74:%.*]] = add nsw i64 [[CONV39]], [[MUL73]]
// CHECK-NEXT:    [[CONV75:%.*]] = trunc i64 [[ADD74]] to i32
// CHECK-NEXT:    store i32 [[CONV75]], ptr [[K]], align 4
// CHECK-NEXT:    [[TMP36:%.*]] = load i32, ptr [[K]], align 4
// CHECK-NEXT:    [[CMP76:%.*]] = icmp slt i32 [[TMP36]], 10
// CHECK-NEXT:    br i1 [[CMP76]], label %[[OMP_BODY_NEXT:.*]], label %[[OMP_BODY_CONTINUE:.*]]
// CHECK:       [[OMP_BODY_NEXT]]:
// CHECK-NEXT:    br label %[[OMP_BODY_CONTINUE]]
// CHECK:       [[OMP_BODY_CONTINUE]]:
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_INC:.*]]
// CHECK:       [[OMP_INNER_FOR_INC]]:
// CHECK-NEXT:    [[TMP37:%.*]] = load i64, ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    [[ADD77:%.*]] = add nsw i64 [[TMP37]], 1
// CHECK-NEXT:    store i64 [[ADD77]], ptr [[DOTOMP_IV]], align 8
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// CHECK:       [[OMP_INNER_FOR_END]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_EXIT:.*]]
// CHECK:       [[OMP_LOOP_EXIT]]:
// CHECK-NEXT:    [[TMP38:%.*]] = load ptr, ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    [[TMP39:%.*]] = load i32, ptr [[TMP38]], align 4
// CHECK-NEXT:    call void @__kmpc_for_static_fini(ptr @[[GLOB1]], i32 [[TMP39]])
// CHECK-NEXT:    ret void
void mixed_loop() {
#pragma omp parallel for collapse(3)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      for (int k = j + 1; k < 10; ++k)
	;
}

// CHECK-LABEL: define internal void @_Z16rectangular_loopv.omp_outlined
// CHECK-SAME: ptr noalias noundef [[DOTGLOBAL_TID_:%.*]], ptr noalias noundef [[DOTBOUND_TID_:%.*]]) #[[ATTR1]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTOMP_IV:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[TMP:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[_TMP1:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[_TMP2:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTOMP_LB:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTOMP_UB:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTOMP_STRIDE:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[DOTOMP_IS_LAST:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[J:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[K:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[DOTBOUND_TID__ADDR]], align 8
// CHECK-NEXT:    store i32 0, ptr [[DOTOMP_LB]], align 4
// CHECK-NEXT:    store i32 999, ptr [[DOTOMP_UB]], align 4
// CHECK-NEXT:    store i32 1, ptr [[DOTOMP_STRIDE]], align 4
// CHECK-NEXT:    store i32 0, ptr [[DOTOMP_IS_LAST]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[TMP0]], align 4
// CHECK-NEXT:    call void @__kmpc_for_static_init_4(ptr @[[GLOB1]], i32 [[TMP1]], i32 34, ptr [[DOTOMP_IS_LAST]], ptr [[DOTOMP_LB]], ptr [[DOTOMP_UB]], ptr [[DOTOMP_STRIDE]], i32 1, i32 1)
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTOMP_UB]], align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[TMP2]], 999
// CHECK-NEXT:    br i1 [[CMP]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// CHECK:       [[COND_TRUE]]:
// CHECK-NEXT:    br label %[[COND_END:.*]]
// CHECK:       [[COND_FALSE]]:
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTOMP_UB]], align 4
// CHECK-NEXT:    br label %[[COND_END]]
// CHECK:       [[COND_END]]:
// CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 999, %[[COND_TRUE]] ], [ [[TMP3]], %[[COND_FALSE]] ]
// CHECK-NEXT:    store i32 [[COND]], ptr [[DOTOMP_UB]], align 4
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, ptr [[DOTOMP_LB]], align 4
// CHECK-NEXT:    store i32 [[TMP4]], ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_COND:.*]]
// CHECK:       [[OMP_INNER_FOR_COND]]:
// CHECK-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, ptr [[DOTOMP_UB]], align 4
// CHECK-NEXT:    [[CMP3:%.*]] = icmp sle i32 [[TMP5]], [[TMP6]]
// CHECK-NEXT:    br i1 [[CMP3]], label %[[OMP_INNER_FOR_BODY:.*]], label %[[OMP_INNER_FOR_END:.*]]
// CHECK:       [[OMP_INNER_FOR_BODY]]:
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[TMP7]], 100
// CHECK-NEXT:    [[MUL:%.*]] = mul nsw i32 [[DIV]], 1
// CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 0, [[MUL]]
// CHECK-NEXT:    store i32 [[ADD]], ptr [[I]], align 4
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    [[DIV4:%.*]] = sdiv i32 [[TMP9]], 100
// CHECK-NEXT:    [[MUL5:%.*]] = mul nsw i32 [[DIV4]], 100
// CHECK-NEXT:    [[SUB:%.*]] = sub nsw i32 [[TMP8]], [[MUL5]]
// CHECK-NEXT:    [[DIV6:%.*]] = sdiv i32 [[SUB]], 10
// CHECK-NEXT:    [[MUL7:%.*]] = mul nsw i32 [[DIV6]], 1
// CHECK-NEXT:    [[ADD8:%.*]] = add nsw i32 0, [[MUL7]]
// CHECK-NEXT:    store i32 [[ADD8]], ptr [[J]], align 4
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    [[DIV9:%.*]] = sdiv i32 [[TMP11]], 100
// CHECK-NEXT:    [[MUL10:%.*]] = mul nsw i32 [[DIV9]], 100
// CHECK-NEXT:    [[SUB11:%.*]] = sub nsw i32 [[TMP10]], [[MUL10]]
// CHECK-NEXT:    [[TMP12:%.*]] = load i32, ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    [[TMP13:%.*]] = load i32, ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    [[DIV12:%.*]] = sdiv i32 [[TMP13]], 100
// CHECK-NEXT:    [[MUL13:%.*]] = mul nsw i32 [[DIV12]], 100
// CHECK-NEXT:    [[SUB14:%.*]] = sub nsw i32 [[TMP12]], [[MUL13]]
// CHECK-NEXT:    [[DIV15:%.*]] = sdiv i32 [[SUB14]], 10
// CHECK-NEXT:    [[MUL16:%.*]] = mul nsw i32 [[DIV15]], 10
// CHECK-NEXT:    [[SUB17:%.*]] = sub nsw i32 [[SUB11]], [[MUL16]]
// CHECK-NEXT:    [[MUL18:%.*]] = mul nsw i32 [[SUB17]], 1
// CHECK-NEXT:    [[ADD19:%.*]] = add nsw i32 0, [[MUL18]]
// CHECK-NEXT:    store i32 [[ADD19]], ptr [[K]], align 4
// CHECK-NEXT:    br label %[[OMP_BODY_CONTINUE:.*]]
// CHECK:       [[OMP_BODY_CONTINUE]]:
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_INC:.*]]
// CHECK:       [[OMP_INNER_FOR_INC]]:
// CHECK-NEXT:    [[TMP14:%.*]] = load i32, ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    [[ADD20:%.*]] = add nsw i32 [[TMP14]], 1
// CHECK-NEXT:    store i32 [[ADD20]], ptr [[DOTOMP_IV]], align 4
// CHECK-NEXT:    br label %[[OMP_INNER_FOR_COND]]
// CHECK:       [[OMP_INNER_FOR_END]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_EXIT:.*]]
// CHECK:       [[OMP_LOOP_EXIT]]:
// CHECK-NEXT:    call void @__kmpc_for_static_fini(ptr @[[GLOB1]], i32 [[TMP1]])
// CHECK-NEXT:    ret void
void rectangular_loop() {
#pragma omp parallel for collapse(3)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      for (int k = 0; k < 10; ++k)
        ;
}
