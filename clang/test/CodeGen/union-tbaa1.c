// RUN: %clang_cc1 %s -triple hexagon-unknown-elf -union-tbaa -O2 -emit-llvm -o - | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 %s -triple hexagon-unknown-elf -O2 -emit-llvm -o - | FileCheck %s --check-prefix=DISABLED

typedef union __attribute__((aligned(4))) {
  unsigned short uh[2];
  unsigned uw;
} vect32;

void bar(vect32 p[][2]);

// ENABLED-LABEL: define dso_local void @fred
// ENABLED-SAME: (i32 noundef [[NUM:%.*]], ptr nocapture noundef writeonly [[VEC:%.*]], ptr nocapture noundef readonly [[INDEX:%.*]], ptr nocapture noundef readonly [[ARR:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// ENABLED-NEXT:  entry:
// ENABLED-NEXT:    [[TMP:%.*]] = alloca [4 x [2 x %union.vect32]], align 8
// ENABLED-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[TMP]]) #[[ATTR3:[0-9]+]]
// ENABLED-NEXT:    [[TMP0:%.*]] = load i32, ptr [[INDEX]], align 4, !tbaa [[TBAA2:![0-9]+]]
// ENABLED-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr [[ARR]], i32 [[TMP0]]
// ENABLED-NEXT:    [[TMP1:%.*]] = load i32, ptr [[ARRAYIDX]], align 4, !tbaa [[TBAA2]]
// ENABLED-NEXT:    [[MUL:%.*]] = mul i32 [[TMP1]], [[NUM]]
// ENABLED-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x [2 x %union.vect32]], ptr [[TMP]], i32 0, i32 [[TMP0]]
// ENABLED-NEXT:    store i32 [[MUL]], ptr [[ARRAYIDX2]], align 8, !tbaa [[TBAA6:![0-9]+]]
// ENABLED-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds [2 x i32], ptr [[ARR]], i32 [[TMP0]], i32 1
// ENABLED-NEXT:    [[TMP2:%.*]] = load i32, ptr [[ARRAYIDX5]], align 4, !tbaa [[TBAA2]]
// ENABLED-NEXT:    [[MUL6:%.*]] = mul i32 [[TMP2]], [[NUM]]
// ENABLED-NEXT:    [[ARRAYIDX8:%.*]] = getelementptr inbounds [4 x [2 x %union.vect32]], ptr [[TMP]], i32 0, i32 [[TMP0]], i32 1
// ENABLED-NEXT:    store i32 [[MUL6]], ptr [[ARRAYIDX8]], align 4, !tbaa [[TBAA6]]
// ENABLED-NEXT:    [[TMP3:%.*]] = lshr i32 [[MUL]], 16
// ENABLED-NEXT:    store i32 [[TMP3]], ptr [[VEC]], align 4, !tbaa [[TBAA2]]
// ENABLED-NEXT:    [[TMP4:%.*]] = load i32, ptr [[INDEX]], align 4, !tbaa [[TBAA2]]
// ENABLED-NEXT:    [[ARRAYIDX14:%.*]] = getelementptr inbounds [4 x [2 x %union.vect32]], ptr [[TMP]], i32 0, i32 [[TMP4]], i32 1
// ENABLED-NEXT:    [[ARRAYIDX15:%.*]] = getelementptr inbounds [2 x i16], ptr [[ARRAYIDX14]], i32 0, i32 1
// ENABLED-NEXT:    [[TMP5:%.*]] = load i16, ptr [[ARRAYIDX15]], align 2, !tbaa [[TBAA8:![0-9]+]]
// ENABLED-NEXT:    [[CONV16:%.*]] = zext i16 [[TMP5]] to i32
// ENABLED-NEXT:    [[ARRAYIDX17:%.*]] = getelementptr inbounds i32, ptr [[VEC]], i32 1
// ENABLED-NEXT:    store i32 [[CONV16]], ptr [[ARRAYIDX17]], align 4, !tbaa [[TBAA2]]
// ENABLED-NEXT:    call void @bar(ptr noundef nonnull [[TMP]]) #[[ATTR3]]
// ENABLED-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[TMP]]) #[[ATTR3]]
// ENABLED-NEXT:    ret void
//
// DISABLED-LABEL: define dso_local void @fred
// DISABLED-SAME: (i32 noundef [[NUM:%.*]], ptr nocapture noundef writeonly [[VEC:%.*]], ptr nocapture noundef readonly [[INDEX:%.*]], ptr nocapture noundef readonly [[ARR:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// DISABLED-NEXT:  entry:
// DISABLED-NEXT:    [[TMP:%.*]] = alloca [4 x [2 x %union.vect32]], align 8
// DISABLED-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[TMP]]) #[[ATTR3:[0-9]+]]
// DISABLED-NEXT:    [[TMP0:%.*]] = load i32, ptr [[INDEX]], align 4, !tbaa [[TBAA2:![0-9]+]]
// DISABLED-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x i32], ptr [[ARR]], i32 [[TMP0]]
// DISABLED-NEXT:    [[TMP1:%.*]] = load i32, ptr [[ARRAYIDX]], align 4, !tbaa [[TBAA2]]
// DISABLED-NEXT:    [[MUL:%.*]] = mul i32 [[TMP1]], [[NUM]]
// DISABLED-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x [2 x %union.vect32]], ptr [[TMP]], i32 0, i32 [[TMP0]]
// DISABLED-NEXT:    store i32 [[MUL]], ptr [[ARRAYIDX2]], align 8, !tbaa [[TBAA6:![0-9]+]]
// DISABLED-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds [2 x i32], ptr [[ARR]], i32 [[TMP0]], i32 1
// DISABLED-NEXT:    [[TMP2:%.*]] = load i32, ptr [[ARRAYIDX5]], align 4, !tbaa [[TBAA2]]
// DISABLED-NEXT:    [[MUL6:%.*]] = mul i32 [[TMP2]], [[NUM]]
// DISABLED-NEXT:    [[ARRAYIDX8:%.*]] = getelementptr inbounds [4 x [2 x %union.vect32]], ptr [[TMP]], i32 0, i32 [[TMP0]], i32 1
// DISABLED-NEXT:    store i32 [[MUL6]], ptr [[ARRAYIDX8]], align 4, !tbaa [[TBAA6]]
// DISABLED-NEXT:    [[TMP3:%.*]] = lshr i32 [[MUL]], 16
// DISABLED-NEXT:    store i32 [[TMP3]], ptr [[VEC]], align 4, !tbaa [[TBAA2]]
// DISABLED-NEXT:    [[TMP4:%.*]] = load i32, ptr [[INDEX]], align 4, !tbaa [[TBAA2]]
// DISABLED-NEXT:    [[ARRAYIDX14:%.*]] = getelementptr inbounds [4 x [2 x %union.vect32]], ptr [[TMP]], i32 0, i32 [[TMP4]], i32 1
// DISABLED-NEXT:    [[ARRAYIDX15:%.*]] = getelementptr inbounds [2 x i16], ptr [[ARRAYIDX14]], i32 0, i32 1
// DISABLED-NEXT:    [[TMP5:%.*]] = load i16, ptr [[ARRAYIDX15]], align 2, !tbaa [[TBAA6]]
// DISABLED-NEXT:    [[CONV16:%.*]] = zext i16 [[TMP5]] to i32
// DISABLED-NEXT:    [[ARRAYIDX17:%.*]] = getelementptr inbounds i32, ptr [[VEC]], i32 1
// DISABLED-NEXT:    store i32 [[CONV16]], ptr [[ARRAYIDX17]], align 4, !tbaa [[TBAA2]]
// DISABLED-NEXT:    call void @bar(ptr noundef nonnull [[TMP]]) #[[ATTR3]]
// DISABLED-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[TMP]]) #[[ATTR3]]
// DISABLED-NEXT:    ret void
//
void fred(unsigned Num, int Vec[2], int *Index, int Arr[4][2]) {
  vect32 Tmp[4][2];
  Tmp[*Index][0].uw = Arr[*Index][0] * Num;
  Tmp[*Index][1].uw = Arr[*Index][1] * Num;
  Vec[0] = Tmp[*Index][0].uh[1];
  Vec[1] = Tmp[*Index][1].uh[1];
  bar(Tmp);
}

// ENABLED-DAG: [[SHORT:![0-9]+]] = !{!"short"
// ENABLED-DAG: [[ANON_UNION:![0-9]+]] = !{!"", [[SHORT]], i64 0, !{{[0-9]+}}, i64 0}
// ENABLED-DAG: [[TBAA8]] = !{[[ANON_UNION]], [[SHORT]], i64 0}
//
// DISABLED-DAG: [[CHAR:![0-9]+]] = !{!"omnipotent char"
// DISABLED-DAG: [[TBAA6]] = !{[[CHAR]], [[CHAR]], i64 0}
