// Regression tests for rectangular #pragma omp tile lowering:
// - Inner loop counter .tile.cnt.* with bound == tile size (constant).
// - Validity guard (icmp + branch) when remainder iterations exist or trip count
//   is not proven divisible at compile time.
//
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

extern "C" void body(int);

// Trip count 6, tile 4 → remainder; body must be guarded.
// IR-LABEL: define dso_local void @_Z18remainder_6_tile_4v(
// IR-SAME: ) #[[ATTR0:[0-9]+]] {
// IR-NEXT:  [[ENTRY:.*:]]
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32 0, ptr [[I]], align 4
// IR-NEXT:    store i32 0, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:       [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 6
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END9:.*]]
// IR:       [[FOR_BODY]]:
// IR-NEXT:    [[TMP1:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP1]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1:.*]]
// IR:       [[FOR_COND1]]:
// IR-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP3]], 4
// IR-NEXT:    [[CMP2:%.*]] = icmp slt i32 6, [[ADD]]
// IR-NEXT:    br i1 [[CMP2]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:       [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:       [[COND_FALSE]]:
// IR-NEXT:    [[TMP4:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD3:%.*]] = add nsw i32 [[TMP4]], 4
// IR-NEXT:    br label %[[COND_END]]
// IR:       [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 6, %[[COND_TRUE]] ], [ [[ADD3]], %[[COND_FALSE]] ]
// IR-NEXT:    [[CMP4:%.*]] = icmp slt i32 [[TMP2]], [[COND]]
// IR-NEXT:    br i1 [[CMP4]], label %[[FOR_BODY5:.*]], label %[[FOR_END:.*]]
// IR:       [[FOR_BODY5]]:
// IR-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul nsw i32 [[TMP5]], 1
// IR-NEXT:    [[ADD6:%.*]] = add nsw i32 0, [[MUL]]
// IR-NEXT:    store i32 [[ADD6]], ptr [[I]], align 4
// IR-NEXT:    [[TMP6:%.*]] = load i32, ptr [[I]], align 4
// IR-NEXT:    call void @body(i32 noundef [[TMP6]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:       [[FOR_INC]]:
// IR-NEXT:    [[TMP7:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP7]], 1
// IR-NEXT:    store i32 [[INC]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1]], !llvm.loop [[LOOP2:![0-9]+]]
// IR:       [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC7:.*]]
// IR:       [[FOR_INC7]]:
// IR-NEXT:    [[TMP8:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD8:%.*]] = add nsw i32 [[TMP8]], 4
// IR-NEXT:    store i32 [[ADD8]], ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop [[LOOP4:![0-9]+]]
// IR:       [[FOR_END9]]:
// IR-NEXT:    ret void
//
void remainder_6_tile_4(void) {
  // Rectangular tile counter and constant tile-size bound.
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    body(i);
}

// Simple stride-1 loop; tile size matches full tiles only (10, tile 5).
// IR-LABEL: define dso_local void @_Z15full_tiles_10_5v(
// IR-SAME: ) #[[ATTR0]] {
// IR-NEXT:  [[ENTRY:.*:]]
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32 0, ptr [[I]], align 4
// IR-NEXT:    store i32 0, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:       [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 10
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END9:.*]]
// IR:       [[FOR_BODY]]:
// IR-NEXT:    [[TMP1:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP1]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1:.*]]
// IR:       [[FOR_COND1]]:
// IR-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP3]], 5
// IR-NEXT:    [[CMP2:%.*]] = icmp slt i32 10, [[ADD]]
// IR-NEXT:    br i1 [[CMP2]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:       [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:       [[COND_FALSE]]:
// IR-NEXT:    [[TMP4:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD3:%.*]] = add nsw i32 [[TMP4]], 5
// IR-NEXT:    br label %[[COND_END]]
// IR:       [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 10, %[[COND_TRUE]] ], [ [[ADD3]], %[[COND_FALSE]] ]
// IR-NEXT:    [[CMP4:%.*]] = icmp slt i32 [[TMP2]], [[COND]]
// IR-NEXT:    br i1 [[CMP4]], label %[[FOR_BODY5:.*]], label %[[FOR_END:.*]]
// IR:       [[FOR_BODY5]]:
// IR-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul nsw i32 [[TMP5]], 1
// IR-NEXT:    [[ADD6:%.*]] = add nsw i32 0, [[MUL]]
// IR-NEXT:    store i32 [[ADD6]], ptr [[I]], align 4
// IR-NEXT:    [[TMP6:%.*]] = load i32, ptr [[I]], align 4
// IR-NEXT:    call void @body(i32 noundef [[TMP6]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:       [[FOR_INC]]:
// IR-NEXT:    [[TMP7:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP7]], 1
// IR-NEXT:    store i32 [[INC]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1]], !llvm.loop [[LOOP5:![0-9]+]]
// IR:       [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC7:.*]]
// IR:       [[FOR_INC7]]:
// IR-NEXT:    [[TMP8:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD8:%.*]] = add nsw i32 [[TMP8]], 5
// IR-NEXT:    store i32 [[ADD8]], ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop [[LOOP6:![0-9]+]]
// IR:       [[FOR_END9]]:
// IR-NEXT:    ret void
//
void full_tiles_10_5(void) {
#pragma omp tile sizes(5)
  for (int i = 0; i < 10; ++i)
    body(i);
}

// Variable tile size: clamp (TS<=0 ? 1 : TS) is inlined; bound compares to %cond.
// IR-LABEL: define dso_local void @_Z8var_tilei(
// IR-SAME: i32 noundef [[TS:%.*]]) #[[ATTR0]] {
// IR-NEXT:  [[ENTRY:.*:]]
// IR-NEXT:    [[TS_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32 [[TS]], ptr [[TS_ADDR]], align 4
// IR-NEXT:    store i32 0, ptr [[I]], align 4
// IR-NEXT:    store i32 0, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:       [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 12
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END24:.*]]
// IR:       [[FOR_BODY]]:
// IR-NEXT:    [[TMP1:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP1]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1:.*]]
// IR:       [[FOR_COND1]]:
// IR-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[TMP4:%.*]] = load i32, ptr [[TS_ADDR]], align 4
// IR-NEXT:    [[CMP2:%.*]] = icmp sle i32 [[TMP4]], 0
// IR-NEXT:    br i1 [[CMP2]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:       [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:       [[COND_FALSE]]:
// IR-NEXT:    [[TMP5:%.*]] = load i32, ptr [[TS_ADDR]], align 4
// IR-NEXT:    br label %[[COND_END]]
// IR:       [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 1, %[[COND_TRUE]] ], [ [[TMP5]], %[[COND_FALSE]] ]
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP3]], [[COND]]
// IR-NEXT:    [[CMP3:%.*]] = icmp slt i32 12, [[ADD]]
// IR-NEXT:    br i1 [[CMP3]], label %[[COND_TRUE4:.*]], label %[[COND_FALSE5:.*]]
// IR:       [[COND_TRUE4]]:
// IR-NEXT:    br label %[[COND_END12:.*]]
// IR:       [[COND_FALSE5]]:
// IR-NEXT:    [[TMP6:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[TMP7:%.*]] = load i32, ptr [[TS_ADDR]], align 4
// IR-NEXT:    [[CMP6:%.*]] = icmp sle i32 [[TMP7]], 0
// IR-NEXT:    br i1 [[CMP6]], label %[[COND_TRUE7:.*]], label %[[COND_FALSE8:.*]]
// IR:       [[COND_TRUE7]]:
// IR-NEXT:    br label %[[COND_END9:.*]]
// IR:       [[COND_FALSE8]]:
// IR-NEXT:    [[TMP8:%.*]] = load i32, ptr [[TS_ADDR]], align 4
// IR-NEXT:    br label %[[COND_END9]]
// IR:       [[COND_END9]]:
// IR-NEXT:    [[COND10:%.*]] = phi i32 [ 1, %[[COND_TRUE7]] ], [ [[TMP8]], %[[COND_FALSE8]] ]
// IR-NEXT:    [[ADD11:%.*]] = add nsw i32 [[TMP6]], [[COND10]]
// IR-NEXT:    br label %[[COND_END12]]
// IR:       [[COND_END12]]:
// IR-NEXT:    [[COND13:%.*]] = phi i32 [ 12, %[[COND_TRUE4]] ], [ [[ADD11]], %[[COND_END9]] ]
// IR-NEXT:    [[CMP14:%.*]] = icmp slt i32 [[TMP2]], [[COND13]]
// IR-NEXT:    br i1 [[CMP14]], label %[[FOR_BODY15:.*]], label %[[FOR_END:.*]]
// IR:       [[FOR_BODY15]]:
// IR-NEXT:    [[TMP9:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul nsw i32 [[TMP9]], 1
// IR-NEXT:    [[ADD16:%.*]] = add nsw i32 0, [[MUL]]
// IR-NEXT:    store i32 [[ADD16]], ptr [[I]], align 4
// IR-NEXT:    [[TMP10:%.*]] = load i32, ptr [[I]], align 4
// IR-NEXT:    call void @body(i32 noundef [[TMP10]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:       [[FOR_INC]]:
// IR-NEXT:    [[TMP11:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP11]], 1
// IR-NEXT:    store i32 [[INC]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1]], !llvm.loop [[LOOP7:![0-9]+]]
// IR:       [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC17:.*]]
// IR:       [[FOR_INC17]]:
// IR-NEXT:    [[TMP12:%.*]] = load i32, ptr [[TS_ADDR]], align 4
// IR-NEXT:    [[CMP18:%.*]] = icmp sle i32 [[TMP12]], 0
// IR-NEXT:    br i1 [[CMP18]], label %[[COND_TRUE19:.*]], label %[[COND_FALSE20:.*]]
// IR:       [[COND_TRUE19]]:
// IR-NEXT:    br label %[[COND_END21:.*]]
// IR:       [[COND_FALSE20]]:
// IR-NEXT:    [[TMP13:%.*]] = load i32, ptr [[TS_ADDR]], align 4
// IR-NEXT:    br label %[[COND_END21]]
// IR:       [[COND_END21]]:
// IR-NEXT:    [[COND22:%.*]] = phi i32 [ 1, %[[COND_TRUE19]] ], [ [[TMP13]], %[[COND_FALSE20]] ]
// IR-NEXT:    [[TMP14:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD23:%.*]] = add nsw i32 [[TMP14]], [[COND22]]
// IR-NEXT:    store i32 [[ADD23]], ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop [[LOOP8:![0-9]+]]
// IR:       [[FOR_END24]]:
// IR-NEXT:    ret void
//
void var_tile(int ts) {
#pragma omp tile sizes(ts)
  for (int i = 0; i < 12; ++i)
    body(i);
}

// Two tiled dimensions → two tile counters.
// IR-LABEL: define dso_local void @_Z7two_dimv(
// IR-SAME: ) #[[ATTR0]] {
// IR-NEXT:  [[ENTRY:.*:]]
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_1_IV_J:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_1_IV_J:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32 0, ptr [[I]], align 4
// IR-NEXT:    store i32 0, ptr [[J]], align 4
// IR-NEXT:    store i32 0, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:       [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 3
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END31:.*]]
// IR:       [[FOR_BODY]]:
// IR-NEXT:    store i32 0, ptr [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND1:.*]]
// IR:       [[FOR_COND1]]:
// IR-NEXT:    [[TMP1:%.*]] = load i32, ptr [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[CMP2:%.*]] = icmp slt i32 [[TMP1]], 3
// IR-NEXT:    br i1 [[CMP2]], label %[[FOR_BODY3:.*]], label %[[FOR_END28:.*]]
// IR:       [[FOR_BODY3]]:
// IR-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP2]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4:.*]]
// IR:       [[FOR_COND4]]:
// IR-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP4:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP4]], 2
// IR-NEXT:    [[CMP5:%.*]] = icmp slt i32 3, [[ADD]]
// IR-NEXT:    br i1 [[CMP5]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:       [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:       [[COND_FALSE]]:
// IR-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD6:%.*]] = add nsw i32 [[TMP5]], 2
// IR-NEXT:    br label %[[COND_END]]
// IR:       [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 3, %[[COND_TRUE]] ], [ [[ADD6]], %[[COND_FALSE]] ]
// IR-NEXT:    [[CMP7:%.*]] = icmp slt i32 [[TMP3]], [[COND]]
// IR-NEXT:    br i1 [[CMP7]], label %[[FOR_BODY8:.*]], label %[[FOR_END25:.*]]
// IR:       [[FOR_BODY8]]:
// IR-NEXT:    [[TMP6:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul nsw i32 [[TMP6]], 1
// IR-NEXT:    [[ADD9:%.*]] = add nsw i32 0, [[MUL]]
// IR-NEXT:    store i32 [[ADD9]], ptr [[I]], align 4
// IR-NEXT:    [[TMP7:%.*]] = load i32, ptr [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    store i32 [[TMP7]], ptr [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND10:.*]]
// IR:       [[FOR_COND10]]:
// IR-NEXT:    [[TMP8:%.*]] = load i32, ptr [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[TMP9:%.*]] = load i32, ptr [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD11:%.*]] = add nsw i32 [[TMP9]], 2
// IR-NEXT:    [[CMP12:%.*]] = icmp slt i32 3, [[ADD11]]
// IR-NEXT:    br i1 [[CMP12]], label %[[COND_TRUE13:.*]], label %[[COND_FALSE14:.*]]
// IR:       [[COND_TRUE13]]:
// IR-NEXT:    br label %[[COND_END16:.*]]
// IR:       [[COND_FALSE14]]:
// IR-NEXT:    [[TMP10:%.*]] = load i32, ptr [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD15:%.*]] = add nsw i32 [[TMP10]], 2
// IR-NEXT:    br label %[[COND_END16]]
// IR:       [[COND_END16]]:
// IR-NEXT:    [[COND17:%.*]] = phi i32 [ 3, %[[COND_TRUE13]] ], [ [[ADD15]], %[[COND_FALSE14]] ]
// IR-NEXT:    [[CMP18:%.*]] = icmp slt i32 [[TMP8]], [[COND17]]
// IR-NEXT:    br i1 [[CMP18]], label %[[FOR_BODY19:.*]], label %[[FOR_END:.*]]
// IR:       [[FOR_BODY19]]:
// IR-NEXT:    [[TMP11:%.*]] = load i32, ptr [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[MUL20:%.*]] = mul nsw i32 [[TMP11]], 1
// IR-NEXT:    [[ADD21:%.*]] = add nsw i32 0, [[MUL20]]
// IR-NEXT:    store i32 [[ADD21]], ptr [[J]], align 4
// IR-NEXT:    [[TMP12:%.*]] = load i32, ptr [[I]], align 4
// IR-NEXT:    [[TMP13:%.*]] = load i32, ptr [[J]], align 4
// IR-NEXT:    [[ADD22:%.*]] = add nsw i32 [[TMP12]], [[TMP13]]
// IR-NEXT:    call void @body(i32 noundef [[ADD22]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:       [[FOR_INC]]:
// IR-NEXT:    [[TMP14:%.*]] = load i32, ptr [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP14]], 1
// IR-NEXT:    store i32 [[INC]], ptr [[DOTTILE_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND10]], !llvm.loop [[LOOP9:![0-9]+]]
// IR:       [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC23:.*]]
// IR:       [[FOR_INC23]]:
// IR-NEXT:    [[TMP15:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC24:%.*]] = add nsw i32 [[TMP15]], 1
// IR-NEXT:    store i32 [[INC24]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND4]], !llvm.loop [[LOOP10:![0-9]+]]
// IR:       [[FOR_END25]]:
// IR-NEXT:    br label %[[FOR_INC26:.*]]
// IR:       [[FOR_INC26]]:
// IR-NEXT:    [[TMP16:%.*]] = load i32, ptr [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    [[ADD27:%.*]] = add nsw i32 [[TMP16]], 2
// IR-NEXT:    store i32 [[ADD27]], ptr [[DOTFLOOR_1_IV_J]], align 4
// IR-NEXT:    br label %[[FOR_COND1]], !llvm.loop [[LOOP11:![0-9]+]]
// IR:       [[FOR_END28]]:
// IR-NEXT:    br label %[[FOR_INC29:.*]]
// IR:       [[FOR_INC29]]:
// IR-NEXT:    [[TMP17:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD30:%.*]] = add nsw i32 [[TMP17]], 2
// IR-NEXT:    store i32 [[ADD30]], ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop [[LOOP12:![0-9]+]]
// IR:       [[FOR_END31]]:
// IR-NEXT:    ret void
//
void two_dim(void) {
#pragma omp tile sizes(2, 2)
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      body(i + j);
}

// Nested loop in tile body: outer nest still uses .tile.cnt for tiled dim.
// IR-LABEL: define dso_local void @_Z14nested_inner_ji(
// IR-SAME: i32 noundef [[N:%.*]]) #[[ATTR0]] {
// IR-NEXT:  [[ENTRY:.*:]]
// IR-NEXT:    [[N_ADDR:%.*]] = alloca i32, align 4
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[J:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32 [[N]], ptr [[N_ADDR]], align 4
// IR-NEXT:    store i32 0, ptr [[I]], align 4
// IR-NEXT:    store i32 0, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:       [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 6
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END16:.*]]
// IR:       [[FOR_BODY]]:
// IR-NEXT:    [[TMP1:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP1]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1:.*]]
// IR:       [[FOR_COND1]]:
// IR-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP3]], 4
// IR-NEXT:    [[CMP2:%.*]] = icmp slt i32 6, [[ADD]]
// IR-NEXT:    br i1 [[CMP2]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:       [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:       [[COND_FALSE]]:
// IR-NEXT:    [[TMP4:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD3:%.*]] = add nsw i32 [[TMP4]], 4
// IR-NEXT:    br label %[[COND_END]]
// IR:       [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 6, %[[COND_TRUE]] ], [ [[ADD3]], %[[COND_FALSE]] ]
// IR-NEXT:    [[CMP4:%.*]] = icmp slt i32 [[TMP2]], [[COND]]
// IR-NEXT:    br i1 [[CMP4]], label %[[FOR_BODY5:.*]], label %[[FOR_END13:.*]]
// IR:       [[FOR_BODY5]]:
// IR-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul nsw i32 [[TMP5]], 1
// IR-NEXT:    [[ADD6:%.*]] = add nsw i32 0, [[MUL]]
// IR-NEXT:    store i32 [[ADD6]], ptr [[I]], align 4
// IR-NEXT:    store i32 0, ptr [[J]], align 4
// IR-NEXT:    br label %[[FOR_COND7:.*]]
// IR:       [[FOR_COND7]]:
// IR-NEXT:    [[TMP6:%.*]] = load i32, ptr [[J]], align 4
// IR-NEXT:    [[TMP7:%.*]] = load i32, ptr [[N_ADDR]], align 4
// IR-NEXT:    [[CMP8:%.*]] = icmp slt i32 [[TMP6]], [[TMP7]]
// IR-NEXT:    br i1 [[CMP8]], label %[[FOR_BODY9:.*]], label %[[FOR_END:.*]]
// IR:       [[FOR_BODY9]]:
// IR-NEXT:    [[TMP8:%.*]] = load i32, ptr [[I]], align 4
// IR-NEXT:    [[TMP9:%.*]] = load i32, ptr [[J]], align 4
// IR-NEXT:    [[ADD10:%.*]] = add nsw i32 [[TMP8]], [[TMP9]]
// IR-NEXT:    call void @body(i32 noundef [[ADD10]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:       [[FOR_INC]]:
// IR-NEXT:    [[TMP10:%.*]] = load i32, ptr [[J]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP10]], 1
// IR-NEXT:    store i32 [[INC]], ptr [[J]], align 4
// IR-NEXT:    br label %[[FOR_COND7]], !llvm.loop [[LOOP13:![0-9]+]]
// IR:       [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC11:.*]]
// IR:       [[FOR_INC11]]:
// IR-NEXT:    [[TMP11:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC12:%.*]] = add nsw i32 [[TMP11]], 1
// IR-NEXT:    store i32 [[INC12]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1]], !llvm.loop [[LOOP14:![0-9]+]]
// IR:       [[FOR_END13]]:
// IR-NEXT:    br label %[[FOR_INC14:.*]]
// IR:       [[FOR_INC14]]:
// IR-NEXT:    [[TMP12:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD15:%.*]] = add nsw i32 [[TMP12]], 4
// IR-NEXT:    store i32 [[ADD15]], ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop [[LOOP15:![0-9]+]]
// IR:       [[FOR_END16]]:
// IR-NEXT:    ret void
//
void nested_inner_j(int n) {
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < n; ++j)
      body(i + j);
}

// Predicate elision: when trip count is evenly divisible by constant tile size,
// no if-guard is needed (no icmp against trip count in tile body).
// IR-LABEL: define dso_local void @_Z16elided_predicatev(
// IR-SAME: ) #[[ATTR0]] {
// IR-NEXT:  [[ENTRY:.*:]]
// IR-NEXT:    [[I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTFLOOR_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    [[DOTTILE_0_IV_I:%.*]] = alloca i32, align 4
// IR-NEXT:    store i32 0, ptr [[I]], align 4
// IR-NEXT:    store i32 0, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND:.*]]
// IR:       [[FOR_COND]]:
// IR-NEXT:    [[TMP0:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP0]], 12
// IR-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END9:.*]]
// IR:       [[FOR_BODY]]:
// IR-NEXT:    [[TMP1:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    store i32 [[TMP1]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1:.*]]
// IR:       [[FOR_COND1]]:
// IR-NEXT:    [[TMP2:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[TMP3:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP3]], 4
// IR-NEXT:    [[CMP2:%.*]] = icmp slt i32 12, [[ADD]]
// IR-NEXT:    br i1 [[CMP2]], label %[[COND_TRUE:.*]], label %[[COND_FALSE:.*]]
// IR:       [[COND_TRUE]]:
// IR-NEXT:    br label %[[COND_END:.*]]
// IR:       [[COND_FALSE]]:
// IR-NEXT:    [[TMP4:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD3:%.*]] = add nsw i32 [[TMP4]], 4
// IR-NEXT:    br label %[[COND_END]]
// IR:       [[COND_END]]:
// IR-NEXT:    [[COND:%.*]] = phi i32 [ 12, %[[COND_TRUE]] ], [ [[ADD3]], %[[COND_FALSE]] ]
// IR-NEXT:    [[CMP4:%.*]] = icmp slt i32 [[TMP2]], [[COND]]
// IR-NEXT:    br i1 [[CMP4]], label %[[FOR_BODY5:.*]], label %[[FOR_END:.*]]
// IR:       [[FOR_BODY5]]:
// IR-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[MUL:%.*]] = mul nsw i32 [[TMP5]], 1
// IR-NEXT:    [[ADD6:%.*]] = add nsw i32 0, [[MUL]]
// IR-NEXT:    store i32 [[ADD6]], ptr [[I]], align 4
// IR-NEXT:    [[TMP6:%.*]] = load i32, ptr [[I]], align 4
// IR-NEXT:    call void @body(i32 noundef [[TMP6]])
// IR-NEXT:    br label %[[FOR_INC:.*]]
// IR:       [[FOR_INC]]:
// IR-NEXT:    [[TMP7:%.*]] = load i32, ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP7]], 1
// IR-NEXT:    store i32 [[INC]], ptr [[DOTTILE_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND1]], !llvm.loop [[LOOP16:![0-9]+]]
// IR:       [[FOR_END]]:
// IR-NEXT:    br label %[[FOR_INC7:.*]]
// IR:       [[FOR_INC7]]:
// IR-NEXT:    [[TMP8:%.*]] = load i32, ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    [[ADD8:%.*]] = add nsw i32 [[TMP8]], 4
// IR-NEXT:    store i32 [[ADD8]], ptr [[DOTFLOOR_0_IV_I]], align 4
// IR-NEXT:    br label %[[FOR_COND]], !llvm.loop [[LOOP17:![0-9]+]]
// IR:       [[FOR_END9]]:
// IR-NEXT:    ret void
//
void elided_predicate(void) {
  // Trip count 12, tile size 4 → 12 % 4 == 0, no predicate.
#pragma omp tile sizes(4)
  for (int i = 0; i < 12; ++i)
    body(i);
}
//.
// IR: [[LOOP2]] = distinct !{[[LOOP2]], [[META3:![0-9]+]]}
// IR: [[META3]] = !{!"llvm.loop.mustprogress"}
// IR: [[LOOP4]] = distinct !{[[LOOP4]], [[META3]]}
// IR: [[LOOP5]] = distinct !{[[LOOP5]], [[META3]]}
// IR: [[LOOP6]] = distinct !{[[LOOP6]], [[META3]]}
// IR: [[LOOP7]] = distinct !{[[LOOP7]], [[META3]]}
// IR: [[LOOP8]] = distinct !{[[LOOP8]], [[META3]]}
// IR: [[LOOP9]] = distinct !{[[LOOP9]], [[META3]]}
// IR: [[LOOP10]] = distinct !{[[LOOP10]], [[META3]]}
// IR: [[LOOP11]] = distinct !{[[LOOP11]], [[META3]]}
// IR: [[LOOP12]] = distinct !{[[LOOP12]], [[META3]]}
// IR: [[LOOP13]] = distinct !{[[LOOP13]], [[META3]]}
// IR: [[LOOP14]] = distinct !{[[LOOP14]], [[META3]]}
// IR: [[LOOP15]] = distinct !{[[LOOP15]], [[META3]]}
// IR: [[LOOP16]] = distinct !{[[LOOP16]], [[META3]]}
// IR: [[LOOP17]] = distinct !{[[LOOP17]], [[META3]]}
//.
