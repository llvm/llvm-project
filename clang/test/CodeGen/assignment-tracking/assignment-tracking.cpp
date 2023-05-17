// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone -O0 \
// RUN:     -emit-llvm  -fexperimental-assignment-tracking=forced %s -o -        \
// RUN:     -disable-O0-optnone                                                  \
// RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

// Based on llvm/test/DebugInfo/Generic/track-assignments.ll - check that using
// -Xclang -fexperimental-assignment-tracking results in emitting (or, as it is
// set up currently, telling llvm to create) assignment tracking metadata.
//
// See the original test for more info.

struct Inner { int A, B; };
struct Outer { Inner A, B; };
struct Large { int A[10]; };
struct LCopyCtor { int A[4]; LCopyCtor(); LCopyCtor(LCopyCtor const &); };
int Value, Index, Cond;
Inner InnerA, InnerB;
Large L;

void zeroInit() { int Z[3] = {0, 0, 0}; }
// CHECK-LABEL: define dso_local void @_Z8zeroInitv
// CHECK:       %Z = alloca [3 x i32], align 4, !DIAssignID ![[ID_0:[0-9]+]]
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_0:[0-9]+]], metadata !DIExpression(), metadata ![[ID_0]], metadata ptr %Z, metadata !DIExpression())
// CHECK:        @llvm.memset{{.*}}, !DIAssignID ![[ID_1:[0-9]+]]
// CHECK-NEXT:   call void @llvm.dbg.assign(metadata i8 0, metadata ![[VAR_0]], metadata !DIExpression(), metadata ![[ID_1]], metadata ptr %Z, metadata !DIExpression())

void memcpyInit() { int A[4] = {0, 1, 2, 3}; }
// CHECK-LABEL: define dso_local void @_Z10memcpyInitv
// CHECK:       %A = alloca [4 x i32], align 16, !DIAssignID ![[ID_2:[0-9]+]]
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_1:[0-9]+]], metadata !DIExpression(), metadata ![[ID_2]], metadata ptr %A, metadata !DIExpression())
// CHECK:        @llvm.memcpy{{.*}}, !DIAssignID ![[ID_3:[0-9]+]]
// CHECK-NEXT:   call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_1]], metadata !DIExpression(), metadata ![[ID_3]], metadata ptr %A, metadata !DIExpression())

void setField() {
  Outer O;
  O.A.B = Value;
}
// CHECK-LABEL: define dso_local void @_Z8setFieldv
// CHECK:       %O = alloca %struct.Outer, align 4, !DIAssignID ![[ID_4:[0-9]+]]
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_2:[0-9]+]], metadata !DIExpression(), metadata ![[ID_4]], metadata ptr %O, metadata !DIExpression())
// CHECK:       store i32 %0, ptr %B, align 4,{{.*}}!DIAssignID ![[ID_5:[0-9]+]]
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i32 %0, metadata ![[VAR_2]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata ![[ID_5]], metadata ptr %B, metadata !DIExpression())

void unknownOffset() {
  int A[2];
  A[Index] = Value;
}
// CHECK-LABEL: define dso_local void @_Z13unknownOffsetv
// CHECK:       %A = alloca [2 x i32], align 4, !DIAssignID ![[ID_6:[0-9]+]]
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_3:[0-9]+]], metadata !DIExpression(), metadata ![[ID_6]], metadata ptr %A, metadata !DIExpression())

Inner sharedAlloca() {
  if (Cond) {
    Inner A = InnerA;
    return A;
  } else {
    Inner B = InnerB;
    return B;
  }
}
// CHECK-LABEL: define dso_local i64 @_Z12sharedAllocav
// CHECK:       %retval = alloca %struct.Inner, align 4, !DIAssignID ![[ID_7:[0-9]+]]
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_4:[0-9]+]], metadata !DIExpression(), metadata ![[ID_7]], metadata ptr %retval, metadata !DIExpression())
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_5:[0-9]+]], metadata !DIExpression(), metadata ![[ID_7]], metadata ptr %retval, metadata !DIExpression())
// CHECK:     if.then:
// CHECK:       call void @llvm.memcpy{{.*}}, !DIAssignID ![[ID_8:[0-9]+]]
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_4]], metadata !DIExpression(), metadata ![[ID_8]], metadata ptr %retval, metadata !DIExpression())
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_5]], metadata !DIExpression(), metadata ![[ID_8]], metadata ptr %retval, metadata !DIExpression())
// CHECK:     if.else:
// CHECK:       call void @llvm.memcpy{{.*}}, !DIAssignID ![[ID_9:[0-9]+]]
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_4]], metadata !DIExpression(), metadata ![[ID_9]], metadata ptr %retval, metadata !DIExpression())
// CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_5]], metadata !DIExpression(), metadata ![[ID_9]], metadata ptr %retval, metadata !DIExpression())

Large sret() {
  Large X = L;
  return X;
}
// CHECK-LABEL: define dso_local void @_Z4sretv
// CHECK:       llvm.dbg.declare

void byval(Large X) {}
// CHECK-LABEL: define dso_local void @_Z5byval5Large
// CHECK:       llvm.dbg.declare

LCopyCtor indirectReturn() {
  LCopyCtor R;
  return R;
}
// CHECK-LABEL: define dso_local void @_Z14indirectReturnv
// CHECK:       call void @llvm.dbg.declare

// CHECK-DAG: ![[VAR_0]] = !DILocalVariable(name: "Z",
// CHECK-DAG: ![[VAR_1]] = !DILocalVariable(name: "A",
// CHECK-DAG: ![[VAR_2]] = !DILocalVariable(name: "O",
// CHECK-DAG: ![[VAR_3]] = !DILocalVariable(name: "A",
// CHECK-DAG: ![[VAR_4]] = !DILocalVariable(name: "B",
// CHECK-DAG: ![[VAR_5]] = !DILocalVariable(name: "A",
