// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

// Constant initializers for data pointers.
extern int external_int;

// CHECK: [[PTRAUTH_G1:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 0, i64 56 }, section "llvm.ptrauth"
// CHECK: @g1 = global ptr [[PTRAUTH_G1]]
int * __ptrauth(1,0,56) g1 = &external_int;

// CHECK: [[PTRAUTH_G2:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 ptrtoint (ptr @g2 to i64), i64 1272 }, section "llvm.ptrauth"
// CHECK: @g2 = global ptr [[PTRAUTH_G2]]
int * __ptrauth(1,1,1272) g2 = &external_int;

// CHECK: @g3 = global ptr null
int * __ptrauth(1,1,871) g3 = 0;

// FIXME: should we make a ptrauth constant for this absolute symbol?
// CHECK: @g4 = global ptr inttoptr (i64 1230 to ptr)
int * __ptrauth(1,1,1902) g4 = (int*) 1230;

// CHECK: [[PTRAUTH_GA0:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 ptrtoint (ptr @ga to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GA1:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 ptrtoint (ptr getelementptr inbounds ([3 x ptr], ptr @ga, i32 0, i32 1) to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GA2:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 ptrtoint (ptr getelementptr inbounds ([3 x ptr], ptr @ga, i32 0, i32 2) to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: @ga = global [3 x ptr] [ptr [[PTRAUTH_GA0]], ptr [[PTRAUTH_GA1]], ptr [[PTRAUTH_GA2]]]
int * __ptrauth(1,1,712) ga[3] = { &external_int, &external_int, &external_int };

struct A {
  int * __ptrauth(1,0,431) f0;
  int * __ptrauth(1,0,9182) f1;
  int * __ptrauth(1,0,783) f2;
};
// CHECK: [[PTRAUTH_GS0:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 0, i64 431 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GS1:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 0, i64 9182 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GS2:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 0, i64 783 }, section "llvm.ptrauth"
// CHECK: @gs1 = global %struct.A { ptr [[PTRAUTH_GS0]], ptr [[PTRAUTH_GS1]], ptr [[PTRAUTH_GS2]] }
struct A gs1 = { &external_int, &external_int, &external_int };

struct B {
  int * __ptrauth(1,1,1276) f0;
  int * __ptrauth(1,1,23674) f1;
  int * __ptrauth(1,1,163) f2;
};
// CHECK: [[PTRAUTH_GS0:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 ptrtoint (ptr @gs2 to i64), i64 1276 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GS1:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 ptrtoint (ptr getelementptr inbounds (%struct.B, ptr @gs2, i32 0, i32 1) to i64), i64 23674 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GS2:@external_int.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_int, i32 1, i64 ptrtoint (ptr getelementptr inbounds (%struct.B, ptr @gs2, i32 0, i32 2) to i64), i64 163 }, section "llvm.ptrauth"
// CHECK: @gs2 = global %struct.B { ptr [[PTRAUTH_GS0]], ptr [[PTRAUTH_GS1]], ptr [[PTRAUTH_GS2]] }
struct B gs2 = { &external_int, &external_int, &external_int };

// Constant initializers for function pointers.
extern void external_function(void);
typedef void (*fpt)(void);

// CHECK: [[PTRAUTH_F1:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 0, i64 56 }, section "llvm.ptrauth"
// CHECK: @f1 = global ptr [[PTRAUTH_F1]]
fpt __ptrauth(1,0,56) f1 = &external_function;

// CHECK: [[PTRAUTH_F2:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 ptrtoint (ptr @f2 to i64), i64 1272 }, section "llvm.ptrauth"
// CHECK: @f2 = global ptr [[PTRAUTH_F2]]
fpt __ptrauth(1,1,1272) f2 = &external_function;

// CHECK: [[PTRAUTH_FA0:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 ptrtoint (ptr @fa to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FA1:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 ptrtoint (ptr getelementptr inbounds ([3 x ptr], ptr @fa, i32 0, i32 1) to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FA2:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 ptrtoint (ptr getelementptr inbounds ([3 x ptr], ptr @fa, i32 0, i32 2) to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: @fa = global [3 x ptr] [ptr [[PTRAUTH_FA0]], ptr [[PTRAUTH_FA1]], ptr [[PTRAUTH_FA2]]]
fpt __ptrauth(1,1,712) fa[3] = { &external_function, &external_function, &external_function };

struct C {
  fpt __ptrauth(1,0,431) f0;
  fpt __ptrauth(1,0,9182) f1;
  fpt __ptrauth(1,0,783) f2;
};
// CHECK: [[PTRAUTH_FS0:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 0, i64 431 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FS1:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 0, i64 9182 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FS2:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 0, i64 783 }, section "llvm.ptrauth"
// CHECK: @fs1 = global %struct.C { ptr [[PTRAUTH_FS0]], ptr [[PTRAUTH_FS1]], ptr [[PTRAUTH_FS2]] }
struct C fs1 = { &external_function, &external_function, &external_function };

struct D {
  fpt __ptrauth(1,1,1276) f0;
  fpt __ptrauth(1,1,23674) f1;
  fpt __ptrauth(1,1,163) f2;
};
// CHECK: [[PTRAUTH_FS0:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 ptrtoint (ptr @fs2 to i64), i64 1276 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FS1:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 ptrtoint (ptr getelementptr inbounds (%struct.D, ptr @fs2, i32 0, i32 1) to i64), i64 23674 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FS2:@external_function.ptrauth.*]] = private constant { ptr, i32, i64, i64 } { ptr @external_function, i32 1, i64 ptrtoint (ptr getelementptr inbounds (%struct.D, ptr @fs2, i32 0, i32 2) to i64), i64 163 }, section "llvm.ptrauth"
// CHECK: @fs2 = global %struct.D { ptr [[PTRAUTH_FS0]], ptr [[PTRAUTH_FS1]], ptr [[PTRAUTH_FS2]] }
struct D fs2 = { &external_function, &external_function, &external_function };
