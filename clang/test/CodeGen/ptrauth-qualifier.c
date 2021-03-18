// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

// Constant initializers for data pointers.
extern int external_int;

// CHECK: [[PTRAUTH_G1:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 0, i64 56 }, section "llvm.ptrauth"
// CHECK: @g1 = global i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_G1]] to i32*)
int * __ptrauth(1,0,56) g1 = &external_int;

// CHECK: [[PTRAUTH_G2:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 ptrtoint (i32** @g2 to i64), i64 1272 }, section "llvm.ptrauth"
// CHECK: @g2 = global i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_G2]] to i32*)
int * __ptrauth(1,1,1272) g2 = &external_int;

// CHECK: @g3 = global i32* null
int * __ptrauth(1,1,871) g3 = 0;

// FIXME: should we make a ptrauth constant for this absolute symbol?
// CHECK: @g4 = global i32* inttoptr (i64 1230 to i32*)
int * __ptrauth(1,1,1902) g4 = (int*) 1230;

// CHECK: [[PTRAUTH_GA0:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 ptrtoint ([3 x i32*]* @ga to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GA1:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 ptrtoint (i32** getelementptr inbounds ([3 x i32*], [3 x i32*]* @ga, i32 0, i32 1) to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GA2:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 ptrtoint (i32** getelementptr inbounds ([3 x i32*], [3 x i32*]* @ga, i32 0, i32 2) to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: @ga = global [3 x i32*] [i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_GA0]] to i32*), i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_GA1]] to i32*), i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_GA2]] to i32*)]
int * __ptrauth(1,1,712) ga[3] = { &external_int, &external_int, &external_int };

struct A {
  int * __ptrauth(1,0,431) f0;
  int * __ptrauth(1,0,9182) f1;
  int * __ptrauth(1,0,783) f2;
};
// CHECK: [[PTRAUTH_GS0:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 0, i64 431 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GS1:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 0, i64 9182 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GS2:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 0, i64 783 }, section "llvm.ptrauth"
// CHECK: @gs1 = global %struct.A { i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_GS0]] to i32*), i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_GS1]] to i32*), i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_GS2]] to i32*) }
struct A gs1 = { &external_int, &external_int, &external_int };

struct B {
  int * __ptrauth(1,1,1276) f0;
  int * __ptrauth(1,1,23674) f1;
  int * __ptrauth(1,1,163) f2;
};
// CHECK: [[PTRAUTH_GS0:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 ptrtoint (%struct.B* @gs2 to i64), i64 1276 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GS1:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 ptrtoint (i32** getelementptr inbounds (%struct.B, %struct.B* @gs2, i32 0, i32 1) to i64), i64 23674 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_GS2:@external_int.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @external_int to i8*), i32 1, i64 ptrtoint (i32** getelementptr inbounds (%struct.B, %struct.B* @gs2, i32 0, i32 2) to i64), i64 163 }, section "llvm.ptrauth"
// CHECK: @gs2 = global %struct.B { i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_GS0]] to i32*), i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_GS1]] to i32*), i32* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_GS2]] to i32*) }
struct B gs2 = { &external_int, &external_int, &external_int };

// Constant initializers for function pointers.
extern void external_function(void);
typedef void (*fpt)(void);

// CHECK: [[PTRAUTH_F1:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 0, i64 56 }, section "llvm.ptrauth"
// CHECK: @f1 = global void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_F1]] to void ()*)
fpt __ptrauth(1,0,56) f1 = &external_function;

// CHECK: [[PTRAUTH_F2:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 ptrtoint (void ()** @f2 to i64), i64 1272 }, section "llvm.ptrauth"
// CHECK: @f2 = global void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_F2]] to void ()*)
fpt __ptrauth(1,1,1272) f2 = &external_function;

// CHECK: [[PTRAUTH_FA0:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 ptrtoint ([3 x void ()*]* @fa to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FA1:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 ptrtoint (void ()** getelementptr inbounds ([3 x void ()*], [3 x void ()*]* @fa, i32 0, i32 1) to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FA2:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 ptrtoint (void ()** getelementptr inbounds ([3 x void ()*], [3 x void ()*]* @fa, i32 0, i32 2) to i64), i64 712 }, section "llvm.ptrauth"
// CHECK: @fa = global [3 x void ()*] [void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_FA0]] to void ()*), void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_FA1]] to void ()*), void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_FA2]] to void ()*)]
fpt __ptrauth(1,1,712) fa[3] = { &external_function, &external_function, &external_function };

struct C {
  fpt __ptrauth(1,0,431) f0;
  fpt __ptrauth(1,0,9182) f1;
  fpt __ptrauth(1,0,783) f2;
};
// CHECK: [[PTRAUTH_FS0:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 0, i64 431 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FS1:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 0, i64 9182 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FS2:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 0, i64 783 }, section "llvm.ptrauth"
// CHECK: @fs1 = global %struct.C { void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_FS0]] to void ()*), void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_FS1]] to void ()*), void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_FS2]] to void ()*) }
struct C fs1 = { &external_function, &external_function, &external_function };

struct D {
  fpt __ptrauth(1,1,1276) f0;
  fpt __ptrauth(1,1,23674) f1;
  fpt __ptrauth(1,1,163) f2;
};
// CHECK: [[PTRAUTH_FS0:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 ptrtoint (%struct.D* @fs2 to i64), i64 1276 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FS1:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 ptrtoint (void ()** getelementptr inbounds (%struct.D, %struct.D* @fs2, i32 0, i32 1) to i64), i64 23674 }, section "llvm.ptrauth"
// CHECK: [[PTRAUTH_FS2:@external_function.ptrauth.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 1, i64 ptrtoint (void ()** getelementptr inbounds (%struct.D, %struct.D* @fs2, i32 0, i32 2) to i64), i64 163 }, section "llvm.ptrauth"
// CHECK: @fs2 = global %struct.D { void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_FS0]] to void ()*), void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_FS1]] to void ()*), void ()* bitcast ({ i8*, i32, i64, i64 }* [[PTRAUTH_FS2]] to void ()*) }
struct D fs2 = { &external_function, &external_function, &external_function };
