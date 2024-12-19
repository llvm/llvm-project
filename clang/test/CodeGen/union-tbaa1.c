// RUN: %clang_cc1 %s -triple hexagon-unknown-elf -O2 -emit-llvm -o - | FileCheck %s

typedef union __attribute__((aligned(4))) {
  unsigned short uh[2];
  unsigned uw;
} vect32;

void bar(vect32 p[][2]);

// CHECK-LABEL: define dso_local void @fred
// CHECK-SAME: (i32 noundef [[NUM:%.*]], ptr nocapture noundef writeonly initializes((0, 8)) [[VEC:%.*]], ptr nocapture noundef readonly [[INDEX:%.*]], ptr nocapture noundef readonly [[ARR:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP:%.*]] = alloca [4 x [2 x %union.vect32]], align 8
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[TMP]]) #[[ATTR3:[0-9]+]]
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[INDEX]], align 4, !tbaa [[TBAA2:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr [[ARR]], i32 [[TMP0]]
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[ARRAYIDX]], align 4, !tbaa [[TBAA2]]
// CHECK-NEXT:    [[MUL:%.*]] = mul i32 [[TMP1]], [[NUM]]
// CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x [2 x %union.vect32]], ptr [[TMP]], i32 0, i32 [[TMP0]]
// CHECK-NEXT:    store i32 [[MUL]], ptr [[ARRAYIDX2]], align 8, !tbaa [[TBAA6:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds [2 x i32], ptr [[ARR]], i32 [[TMP0]], i32 1
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[ARRAYIDX5]], align 4, !tbaa [[TBAA2]]
// CHECK-NEXT:    [[MUL6:%.*]] = mul i32 [[TMP2]], [[NUM]]
// CHECK-NEXT:    [[ARRAYIDX8:%.*]] = getelementptr inbounds [4 x [2 x %union.vect32]], ptr [[TMP]], i32 0, i32 [[TMP0]], i32 1
// CHECK-NEXT:    store i32 [[MUL6]], ptr [[ARRAYIDX8]], align 4, !tbaa [[TBAA6]]
// CHECK-NEXT:    [[TMP3:%.*]] = lshr i32 [[MUL]], 16
// CHECK-NEXT:    store i32 [[TMP3]], ptr [[VEC]], align 4, !tbaa [[TBAA2]]
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, ptr [[INDEX]], align 4, !tbaa [[TBAA2]]
// CHECK-NEXT:    [[ARRAYIDX14:%.*]] = getelementptr inbounds [4 x [2 x %union.vect32]], ptr [[TMP]], i32 0, i32 [[TMP4]], i32 1
// CHECK-NEXT:    [[ARRAYIDX15:%.*]] = getelementptr inbounds nuw i8, ptr [[ARRAYIDX14]], i32 2
// CHECK-NEXT:    [[TMP5:%.*]] = load i16, ptr [[ARRAYIDX15]], align 2, !tbaa [[TBAA6]]
// CHECK-NEXT:    [[CONV16:%.*]] = zext i16 [[TMP5]] to i32
// CHECK-NEXT:    [[ARRAYIDX17:%.*]] = getelementptr inbounds nuw i8, ptr [[VEC]], i32 4
// CHECK-NEXT:    store i32 [[CONV16]], ptr [[ARRAYIDX17]], align 4, !tbaa [[TBAA2]]
// CHECK-NEXT:    call void @bar(ptr noundef nonnull [[TMP]]) #[[ATTR3]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[TMP]]) #[[ATTR3]]
// CHECK-NEXT:    ret void
//
void fred(unsigned Num, int Vec[2], int *Index, int Arr[4][2]) {
  vect32 Tmp[4][2];
  Tmp[*Index][0].uw = Arr[*Index][0] * Num;
  Tmp[*Index][1].uw = Arr[*Index][1] * Num;
  Vec[0] = Tmp[*Index][0].uh[1];
  Vec[1] = Tmp[*Index][1].uh[1];
  bar(Tmp);
}

// CHECK-DAG: [[CHAR:![0-9]+]] = !{!"omnipotent char"
// CHECK-DAG: [[TBAA6]] = !{[[CHAR]], [[CHAR]], i64 0}
