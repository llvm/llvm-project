// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -o - %s | FileCheck -check-prefixes=CHECK,NODEBUG %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -debug-info-kind=limited -o - %s | FileCheck -check-prefixes=CHECK %s

// CHECK: %[[STRUCT_BASE0:.*]] = type { i32 (...)** }
// CHECK: %[[STRUCT_DERIVED0:.*]] = type { %[[STRUCT_BASE0]] }
// CHECK: %[[STRUCT_A0:.*]] = type { [4 x i32] }
// CHECK: %[[STRUCT_A1:.*]] = type { [8 x i32] }
// CHECK: %[[STRUCT_TRIVIALS:.*]] = type { [4 x i32] }
// CHECK: %[[STRUCT_BASE1:.*]] = type { i32 (...)** }
// CHECK: %[[STRUCT_DERIVED1:.*]] = type { %[[STRUCT_BASE0]], %[[STRUCT_BASE1]] }

// CHECK: @_ZN5Base011nonvirtual0Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*)* @_ZN5Base011nonvirtual0Ev to i8*), i32 0, i64 0, i64 [[TYPEDISC0:22163]] }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base08virtual1Ev_vfpthunk_.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*)* @_ZN5Base08virtual1Ev_vfpthunk_ to i8*), i32 0, i64 0, i64 [[TYPEDISC0]] }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base08virtual3Ev_vfpthunk_.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*)* @_ZN5Base08virtual3Ev_vfpthunk_ to i8*), i32 0, i64 0, i64 [[TYPEDISC0]] }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base016virtual_variadicEiz_vfpthunk_.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*, i32, ...)* @_ZN5Base016virtual_variadicEiz_vfpthunk_ to i8*), i32 0, i64 0, i64 34368 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base011nonvirtual0Ev.ptrauth.1 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*)* @_ZN5Base011nonvirtual0Ev to i8*), i32 0, i64 0, i64 [[TYPEDISC1:35591]] }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base08virtual1Ev_vfpthunk_.ptrauth.2 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*)* @_ZN5Base08virtual1Ev_vfpthunk_ to i8*), i32 0, i64 0, i64 [[TYPEDISC1]] }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base08virtual3Ev_vfpthunk_.ptrauth.3 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*)* @_ZN5Base08virtual3Ev_vfpthunk_ to i8*), i32 0, i64 0, i64 [[TYPEDISC1]] }, section "llvm.ptrauth", align 8
// CHECK: @_ZN8Derived011nonvirtual5Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_DERIVED0]]*)* @_ZN8Derived011nonvirtual5Ev to i8*), i32 0, i64 0, i64 [[TYPEDISC1]] }, section "llvm.ptrauth", align 8
// CHECK: @_ZN8Derived08virtual6Ev_vfpthunk_.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_DERIVED0]]*)* @_ZN8Derived08virtual6Ev_vfpthunk_ to i8*), i32 0, i64 0, i64 [[TYPEDISC1]] }, section "llvm.ptrauth", align 8
// CHECK: @_ZN8Derived010return_aggEv_vfpthunk_.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast ([2 x i64] (%[[STRUCT_DERIVED0]]*)* @_ZN8Derived010return_aggEv_vfpthunk_ to i8*), i32 0, i64 0, i64 64418 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN8Derived04sretEv_vfpthunk_.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_A1]]*, %[[STRUCT_DERIVED0]]*)* @_ZN8Derived04sretEv_vfpthunk_ to i8*), i32 0, i64 0, i64 28187 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN8Derived011trivial_abiE8TrivialS_vfpthunk_.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_DERIVED0]]*, [2 x i64])* @_ZN8Derived011trivial_abiE8TrivialS_vfpthunk_ to i8*), i32 0, i64 0, i64 8992 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base18virtual7Ev_vfpthunk_.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE1]]*)* @_ZN5Base18virtual7Ev_vfpthunk_ to i8*), i32 0, i64 0, i64 [[TYPEDISC2:61596]] }, section "llvm.ptrauth", align 8
// CHECK: @_ZN8Derived18virtual7Ev_vfpthunk_.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_DERIVED1]]*)* @_ZN8Derived18virtual7Ev_vfpthunk_ to i8*), i32 0, i64 0, i64 25206 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base08virtual1Ev_vfpthunk_.ptrauth.4 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*)* @_ZN5Base08virtual1Ev_vfpthunk_ to i8*), i32 0, i64 0, i64 25206 }, section "llvm.ptrauth", align 8

// CHECK: @gmethod0 = global { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base011nonvirtual0Ev.ptrauth.1 to i64), i64 0 }, align 8
// CHECK: @_ZN8Derived011nonvirtual5Ev.ptrauth.6 = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%struct.Derived0*)* @_ZN8Derived011nonvirtual5Ev to i8*), i32 0, i64 0, i64 [[TYPEDISC0]] }, section "llvm.ptrauth", align 8
// CHECK: @gmethod1 = global { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN8Derived011nonvirtual5Ev.ptrauth.6 to i64), i64 0 }, align 8
// CHECK: @gmethod2 = global { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base08virtual1Ev_vfpthunk_.ptrauth to i64), i64 0 }, align 8

// CHECK: @_ZTV5Base0 = unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI5Base0 to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN5Base08virtual1Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN5Base08virtual3Ev.ptrauth to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN5Base016virtual_variadicEiz.ptrauth to i8*)] }, align 8
// CHECK: @_ZN5Base08virtual1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*)* @_ZN5Base08virtual1Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV5Base0, i32 0, i32 0, i32 2) to i64), i64 55600 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base08virtual3Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*)* @_ZN5Base08virtual3Ev to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV5Base0, i32 0, i32 0, i32 3) to i64), i64 53007 }, section "llvm.ptrauth", align 8
// CHECK: @_ZN5Base016virtual_variadicEiz.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%[[STRUCT_BASE0]]*, i32, ...)* @_ZN5Base016virtual_variadicEiz to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV5Base0, i32 0, i32 0, i32 4) to i64), i64 7464 }, section "llvm.ptrauth", align 8

struct Base0 {
  void nonvirtual0();
  virtual void virtual1();
  virtual void virtual3();
  virtual void virtual_variadic(int, ...);
};

struct A0 {
  int d[4];
};

struct A1 {
  int d[8];
};

struct __attribute__((trivial_abi)) TrivialS {
  TrivialS(const TrivialS &);
  ~TrivialS();
  int p[4];
};

struct Derived0 : Base0 {
  void virtual1() override;
  void nonvirtual5();
  virtual void virtual6();
  virtual A0 return_agg();
  virtual A1 sret();
  virtual void trivial_abi(TrivialS);
};

struct Base1 {
  virtual void virtual7();
};

struct Derived1 : Base0, Base1 {
  void virtual1() override;
  void virtual7() override;
};

typedef void (Base0::*MethodTy0)();
typedef void (Base0::*VariadicMethodTy0)(int, ...);
typedef void (Derived0::*MethodTy1)();

// CHECK: define void @_ZN5Base08virtual1Ev(

// CHECK: define void @_Z5test0v()
// CHECK: %[[METHOD0:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[VARMETHOD1:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD2:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD3:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD4:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD5:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD6:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD7:.*]] = alloca { i64, i64 }, align 8
// CHECK: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base011nonvirtual0Ev.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD0]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base08virtual1Ev_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD0]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base08virtual3Ev_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD0]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base016virtual_variadicEiz_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[VARMETHOD1]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base011nonvirtual0Ev.ptrauth.1 to i64), i64 0 }, { i64, i64 }* %[[METHOD2]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base08virtual1Ev_vfpthunk_.ptrauth.2 to i64), i64 0 }, { i64, i64 }* %[[METHOD2]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base08virtual3Ev_vfpthunk_.ptrauth.3 to i64), i64 0 }, { i64, i64 }* %[[METHOD2]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN8Derived011nonvirtual5Ev.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD2]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN8Derived08virtual6Ev_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD2]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN8Derived010return_aggEv_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD3]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN8Derived04sretEv_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD4]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN8Derived011trivial_abiE8TrivialS_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD5]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base18virtual7Ev_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD6]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN8Derived18virtual7Ev_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD7]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base08virtual1Ev_vfpthunk_.ptrauth.4 to i64), i64 0 }, { i64, i64 }* %[[METHOD7]], align 8
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @_ZN5Base08virtual1Ev_vfpthunk_(%[[STRUCT_BASE0]]* nonnull dereferenceable(8) %[[THIS:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca %[[STRUCT_BASE0]]*, align 8
// CHECK: store %[[STRUCT_BASE0]]* %[[THIS]], %[[STRUCT_BASE0]]** %[[THIS_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load %[[STRUCT_BASE0]]*, %[[STRUCT_BASE0]]** %[[THIS_ADDR]], align 8
// CHECK-NEXT: %[[V0:.*]] = load %[[STRUCT_BASE0]]*, %[[STRUCT_BASE0]]** %[[THIS_ADDR]], align 8
// CHECK-NEXT: %[[V1:.*]] = bitcast %[[STRUCT_BASE0]]* %[[THIS1]] to void (%[[STRUCT_BASE0]]*)***
// CHECK-NEXT: %[[VTABLE:.*]] = load void (%[[STRUCT_BASE0]]*)**, void (%[[STRUCT_BASE0]]*)*** %[[V1]], align 8
// CHECK-NEXT: %[[V2:.*]] = ptrtoint void (%[[STRUCT_BASE0]]*)** %[[VTABLE]] to i64
// CHECK-NEXT: %[[V3:.*]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[V2]], i32 2, i64 0)
// CHECK-NEXT: %[[V4:.*]] = inttoptr i64 %[[V3]] to void (%[[STRUCT_BASE0]]*)**
// CHECK-NEXT: %[[VFN:.*]] = getelementptr inbounds void (%[[STRUCT_BASE0]]*)*, void (%[[STRUCT_BASE0]]*)** %[[V4]], i64 0
// CHECK-NEXT: %[[V5:.*]] = load void (%[[STRUCT_BASE0]]*)*, void (%[[STRUCT_BASE0]]*)** %[[VFN]], align 8
// CHECK-NEXT: %[[V6:.*]] = ptrtoint void (%[[STRUCT_BASE0]]*)** %[[VFN]] to i64
// CHECK-NEXT: %[[V7:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V6]], i64 55600)
// CHECK-NEXT: musttail call void %[[V5]](%[[STRUCT_BASE0]]* nonnull dereferenceable(8) %[[V0]]) [ "ptrauth"(i32 0, i64 %[[V7]]) ]
// CHECK-NEXT: ret void

// CHECK: define linkonce_odr hidden void @_ZN5Base08virtual3Ev_vfpthunk_(%[[STRUCT_BASE0]]* nonnull dereferenceable(8) %{{.*}})
// CHECK: %[[VTABLE:.*]] = load void (%[[STRUCT_BASE0]]*)**, void (%[[STRUCT_BASE0]]*)*** %{{.*}}, align 8
// CHECK: %[[V2:.*]] = ptrtoint void (%[[STRUCT_BASE0]]*)** %[[VTABLE]] to i64
// CHECK: %[[V3:.*]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[V2]], i32 2, i64 0)
// CHECK: %[[V4:.*]] = inttoptr i64 %[[V3]] to void (%[[STRUCT_BASE0]]*)**
// CHECK: getelementptr inbounds void (%[[STRUCT_BASE0]]*)*, void (%[[STRUCT_BASE0]]*)** %[[V4]], i64 1
// CHECK: call i64 @llvm.ptrauth.blend.i64(i64 %{{.*}}, i64 53007)

// CHECK: define linkonce_odr hidden void @_ZN5Base016virtual_variadicEiz_vfpthunk_(%[[STRUCT_BASE0]]* nonnull dereferenceable(8) %[[THIS:.*]], i32 %0, ...)
// CHECK: %[[THIS_ADDR:.*]] = alloca %[[STRUCT_BASE0]]*, align 8
// CHECK-NEXT: %[[_ADDR:.*]] = alloca i32, align 4
// CHECK-NEXT: store %[[STRUCT_BASE0]]* %[[THIS]], %[[STRUCT_BASE0]]** %[[THIS_ADDR]], align 8
// CHECK: store i32 %0, i32* %[[_ADDR]], align 4
// CHECK: %[[THIS1:.*]] = load %[[STRUCT_BASE0]]*, %[[STRUCT_BASE0]]** %[[THIS_ADDR]], align 8
// CHECK-NEXT: %[[V1:.*]] = load %[[STRUCT_BASE0]]*, %[[STRUCT_BASE0]]** %[[THIS_ADDR]], align 8
// CHECK-NEXT: %[[V2:.*]] = load i32, i32* %[[_ADDR]], align 4
// CHECK-NEXT: %[[V3:.*]] = bitcast %[[STRUCT_BASE0]]* %[[THIS1]] to void (%[[STRUCT_BASE0]]*, i32, ...)***
// CHECK-NEXT: %[[VTABLE:.*]] = load void (%[[STRUCT_BASE0]]*, i32, ...)**, void (%[[STRUCT_BASE0]]*, i32, ...)*** %[[V3]], align 8
// CHECK-NEXT: %[[V4:.*]] = ptrtoint void (%[[STRUCT_BASE0]]*, i32, ...)** %[[VTABLE]] to i64
// CHECK-NEXT: %[[V5:.*]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[V4]], i32 2, i64 0)
// CHECK-NEXT: %[[V6:.*]] = inttoptr i64 %[[V5]] to void (%[[STRUCT_BASE0]]*, i32, ...)**
// CHECK-NEXT: %[[VFN:.*]] = getelementptr inbounds void (%[[STRUCT_BASE0]]*, i32, ...)*, void (%[[STRUCT_BASE0]]*, i32, ...)** %[[V6]], i64 2
// CHECK-NEXT: %[[V7:.*]] = load void (%[[STRUCT_BASE0]]*, i32, ...)*, void (%[[STRUCT_BASE0]]*, i32, ...)** %[[VFN]], align 8
// CHECK-NEXT: %[[V8:.*]] = ptrtoint void (%[[STRUCT_BASE0]]*, i32, ...)** %[[VFN]] to i64
// CHECK-NEXT: %[[V9:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V8]], i64 7464)
// CHECK-NEXT: musttail call void (%[[STRUCT_BASE0]]*, i32, ...) %[[V7]](%[[STRUCT_BASE0]]* nonnull dereferenceable(8) %[[V1]], i32 %[[V2]], ...) [ "ptrauth"(i32 0, i64 %[[V9]]) ]
// CHECK-NEXT: ret void

// CHECK: define linkonce_odr hidden void @_ZN8Derived08virtual6Ev_vfpthunk_(%[[STRUCT_DERIVED0]]* nonnull dereferenceable(8) %{{.*}})
// CHECK: %[[VTABLE:.*]] = load void (%[[STRUCT_DERIVED0]]*)**, void (%[[STRUCT_DERIVED0]]*)*** %[[V1]], align 8
// CHECK: %[[V2:.*]] = ptrtoint void (%[[STRUCT_DERIVED0]]*)** %[[VTABLE]] to i64
// CHECK: %[[V3:.*]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[V2]], i32 2, i64 0)
// CHECK: %[[V4:.*]] = inttoptr i64 %[[V3]] to void (%[[STRUCT_DERIVED0]]*)**
// CHECK: getelementptr inbounds void (%[[STRUCT_DERIVED0]]*)*, void (%[[STRUCT_DERIVED0]]*)** %[[V4]], i64 3
// CHECK: call i64 @llvm.ptrauth.blend.i64(i64 %{{.*}}, i64 55535)

// Check that the return value of the musttail call isn't copied to a temporary.

// CHECK: define linkonce_odr hidden [2 x i64] @_ZN8Derived010return_aggEv_vfpthunk_(%[[STRUCT_DERIVED0]]* nonnull dereferenceable(8) %{{.*}})
// CHECK: %[[CALL:.*]] = musttail call [2 x i64] %{{.*}}(%[[STRUCT_DERIVED0]]* nonnull dereferenceable(8) %{{.*}}) [ "ptrauth"(i32 0, i64 %{{.*}}) ]
// CHECK-NEXT: ret [2 x i64] %[[CALL]]

// Check that the sret pointer passed to the caller is forwarded to the musttail
// call.

// CHECK: define linkonce_odr hidden void @_ZN8Derived04sretEv_vfpthunk_(%[[STRUCT_A1]]* noalias sret(%struct.A1) align 4 %[[AGG_RESULT:.*]], %[[STRUCT_DERIVED0]]* nonnull dereferenceable(8) %{{.*}})
// CHECK: musttail call void %{{.*}}(%[[STRUCT_A1]]* sret(%struct.A1) align 4 %[[AGG_RESULT]], %[[STRUCT_DERIVED0]]* nonnull dereferenceable(8) %{{.*}}) [ "ptrauth"(i32 0, i64 %{{.*}}) ]
// CHECK-NEXT: ret void

// Check that the thunk function doesn't destruct the trivial_abi argument.

// CHECK: define linkonce_odr hidden void @_ZN8Derived011trivial_abiE8TrivialS_vfpthunk_(%[[STRUCT_DERIVED0]]* nonnull dereferenceable(8) %{{.*}}, [2 x i64] %{{.*}})
// NODEBUG-NOT: call
// CHECK: call i64 @llvm.ptrauth.auth.i64(
// NODEBUG-NOT: call
// CHECK: call i64 @llvm.ptrauth.blend.i64(
// NODEBUG-NOT: call
// CHECK: musttail call void
// CHECK-NEXT: ret void

// CHECK: define linkonce_odr hidden void @_ZN5Base18virtual7Ev_vfpthunk_(%[[STRUCT_BASE1]]* nonnull dereferenceable(8) %{{.*}})
// CHECK: %[[VTABLE:.*]] = load void (%[[STRUCT_BASE1]]*)**, void (%[[STRUCT_BASE1]]*)*** %{{.*}}, align 8
// CHECK: %[[V2:.*]] = ptrtoint void (%[[STRUCT_BASE1]]*)** %[[VTABLE]] to i64
// CHECK: %[[V3:.*]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[V2]], i32 2, i64 0)
// CHECK: %[[V4:.*]] = inttoptr i64 %[[V3]] to void (%[[STRUCT_BASE1]]*)**
// CHECK: getelementptr inbounds void (%[[STRUCT_BASE1]]*)*, void (%[[STRUCT_BASE1]]*)** %[[V4]], i64 0

// CHECK: define linkonce_odr hidden void @_ZN8Derived18virtual7Ev_vfpthunk_(%[[STRUCT_DERIVED1]]* nonnull dereferenceable(16) %{{.*}})
// CHECK: %[[VTABLE:.*]] = load void (%[[STRUCT_DERIVED1]]*)**, void (%[[STRUCT_DERIVED1]]*)*** %[[V1]], align 8
// CHECK: %[[V2:.*]] = ptrtoint void (%[[STRUCT_DERIVED1]]*)** %[[VTABLE]] to i64
// CHECK: %[[V3:.*]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[V2]], i32 2, i64 0)
// CHECK: %[[V4:.*]] = inttoptr i64 %[[V3]] to void (%[[STRUCT_DERIVED1]]*)**
// CHECK: getelementptr inbounds void (%[[STRUCT_DERIVED1]]*)*, void (%[[STRUCT_DERIVED1]]*)** %[[V4]], i64 3

void Base0::virtual1() {}

void test0() {
  MethodTy0 method0;
  method0 = &Base0::nonvirtual0;
  method0 = &Base0::virtual1;
  method0 = &Base0::virtual3;

  VariadicMethodTy0 varmethod1;
  varmethod1 = &Base0::virtual_variadic;

  MethodTy1 method2;
  method2 = &Derived0::nonvirtual0;
  method2 = &Derived0::virtual1;
  method2 = &Derived0::virtual3;
  method2 = &Derived0::nonvirtual5;
  method2 = &Derived0::virtual6;

  A0 (Derived0::*method3)();
  method3 = &Derived0::return_agg;

  A1 (Derived0::*method4)();
  method4 = &Derived0::sret;

  void (Derived0::*method5)(TrivialS);
  method5 = &Derived0::trivial_abi;

  void (Base1::*method6)();
  method6 = &Base1::virtual7;

  void (Derived1::*method7)();
  method7 = &Derived1::virtual7;
  method7 = &Derived1::virtual1;
}

// CHECK: define void @_Z5test1P5Base0MS_FvvE(%[[STRUCT_BASE0]]* %[[A0:.*]], [2 x i64] %[[A1_COERCE:.*]])
// CHECK: %[[A1:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[A0_ADDR:.*]] = alloca %[[STRUCT_BASE0]]*, align 8
// CHECK: %[[A1_ADDR:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[V0:.*]] = bitcast { i64, i64 }* %[[A1]] to [2 x i64]*
// CHECK: store [2 x i64] %[[A1_COERCE]], [2 x i64]* %[[V0]], align 8
// CHECK: %[[A11:.*]] = load { i64, i64 }, { i64, i64 }* %[[A1]], align 8
// CHECK: store %[[STRUCT_BASE0]]* %[[A0]], %[[STRUCT_BASE0]]** %[[A0_ADDR]], align 8
// CHECK: store { i64, i64 } %[[A11]], { i64, i64 }* %[[A1_ADDR]], align 8
// CHECK: %[[V1:.*]] = load %[[STRUCT_BASE0]]*, %[[STRUCT_BASE0]]** %[[A0_ADDR]], align 8
// CHECK: %[[V2:.*]] = load { i64, i64 }, { i64, i64 }* %[[A1_ADDR]], align 8
// CHECK: %[[MEMPTR_ADJ:.*]] = extractvalue { i64, i64 } %[[V2]], 1
// CHECK: %[[MEMPTR_ADJ_SHIFTED:.*]] = ashr i64 %[[MEMPTR_ADJ]], 1
// CHECK: %[[V3:.*]] = bitcast %[[STRUCT_BASE0]]* %[[V1]] to i8*
// CHECK: %[[V4:.*]] = getelementptr inbounds i8, i8* %[[V3]], i64 %[[MEMPTR_ADJ_SHIFTED]]
// CHECK: %[[THIS_ADJUSTED:.*]] = bitcast i8* %[[V4]] to %[[STRUCT_BASE0]]*
// CHECK: %[[MEMPTR_PTR:.*]] = extractvalue { i64, i64 } %[[V2]], 0
// CHECK: %[[V5:.*]] = and i64 %[[MEMPTR_ADJ]], 1
// CHECK: %[[MEMPTR_ISVIRTUAL:.*]] = icmp ne i64 %[[V5]], 0
// CHECK: br i1 %[[MEMPTR_ISVIRTUAL]]

// CHECK: %[[V6:.*]] = bitcast %[[STRUCT_BASE0]]* %[[THIS_ADJUSTED]] to i8**
// CHECK: %[[VTABLE:.*]] = load i8*, i8** %[[V6]], align 8
// CHECK: %[[V7:.*]] = ptrtoint i8* %[[VTABLE]] to i64
// CHECK: %[[V8:.*]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[V7]], i32 2, i64 0)
// CHECK: %[[V9:.*]] = inttoptr i64 %[[V8]] to i8*
// CHECK: %[[V10:.*]] = trunc i64 %[[MEMPTR_PTR]] to i32
// CHECK: %[[V11:.*]] = zext i32 %[[V10]] to i64
// CHECK: %[[V12:.*]] = getelementptr i8, i8* %[[V9]], i64 %[[V11]]
// CHECK: %[[V13:.*]] = bitcast i8* %[[V12]] to void (%[[STRUCT_BASE0]]*)**
// CHECK: %[[MEMPTR_VIRTUALFN:.*]] = load void (%[[STRUCT_BASE0]]*)*, void (%[[STRUCT_BASE0]]*)** %[[V13]], align 8
// CHECK: br

// CHECK: %[[MEMPTR_NONVIRTUALFN:.*]] = inttoptr i64 %[[MEMPTR_PTR]] to void (%[[STRUCT_BASE0]]*)*
// CHECK: br

// CHECK: %[[V14:.*]] = phi void (%[[STRUCT_BASE0]]*)* [ %[[MEMPTR_VIRTUALFN]], {{.*}} ], [ %[[MEMPTR_NONVIRTUALFN]], {{.*}} ]
// CHECK: %[[V15:.*]] = phi i64 [ 0, {{.*}} ], [ [[TYPEDISC0]], {{.*}} ]
// CHECK: call void %[[V14]](%[[STRUCT_BASE0]]* nonnull dereferenceable(8) %[[THIS_ADJUSTED]]) [ "ptrauth"(i32 0, i64 %[[V15]]) ]
// CHECK: ret void

void test1(Base0 *a0, MethodTy0 a1) {
  (a0->*a1)();
}

// CHECK: define void @_Z15testConversion0M5Base0FvvEM8Derived0FvvE([2 x i64] %[[METHOD0_COERCE:.*]], [2 x i64] %[[METHOD1_COERCE:.*]])
// CHECK: %[[METHOD0:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[METHOD1:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[METHOD0_ADDR:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[METHOD1_ADDR:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[V0:.*]] = bitcast { i64, i64 }* %[[METHOD0]] to [2 x i64]*
// CHECK: store [2 x i64] %[[METHOD0_COERCE]], [2 x i64]* %[[V0]], align 8
// CHECK: %[[METHOD01:.*]] = load { i64, i64 }, { i64, i64 }* %[[METHOD0]], align 8
// CHECK: %[[V1:.*]] = bitcast { i64, i64 }* %[[METHOD1]] to [2 x i64]*
// CHECK: store [2 x i64] %[[METHOD1_COERCE]], [2 x i64]* %[[V1]], align 8
// CHECK: %[[METHOD12:.*]] = load { i64, i64 }, { i64, i64 }* %[[METHOD1]], align 8
// CHECK: store { i64, i64 } %[[METHOD01]], { i64, i64 }* %[[METHOD0_ADDR]], align 8
// CHECK: store { i64, i64 } %[[METHOD12]], { i64, i64 }* %[[METHOD1_ADDR]], align 8
// CHECK: %[[V2:.*]] = load { i64, i64 }, { i64, i64 }* %[[METHOD0_ADDR]], align 8
// CHECK: %[[MEMPTR_PTR:.*]] = extractvalue { i64, i64 } %[[V2]], 0
// CHECK: %[[MEMPTR_ADJ:.*]] = extractvalue { i64, i64 } %[[V2]], 1
// CHECK: %[[V3:.*]] = and i64 %[[MEMPTR_ADJ]], 1
// CHECK: %[[IS_VIRTUAL_OFFSET:.*]] = icmp ne i64 %[[V3]], 0
// CHECK: br i1 %[[IS_VIRTUAL_OFFSET]]

// CHECK: %[[V4:.*]] = inttoptr i64 %[[MEMPTR_PTR]] to i8*
// CHECK: %[[V5:.*]] = icmp ne i8* %[[V4]], null
// CHECK: br i1 %[[V5]]

// CHECK: %[[V6:.*]] = ptrtoint i8* %[[V4]] to i64
// CHECK: %[[V7:.*]] = call i64 @llvm.ptrauth.resign.i64(i64 %[[V6]], i32 0, i64 [[TYPEDISC0]], i32 0, i64 [[TYPEDISC1]])
// CHECK: %[[V8:.*]] = inttoptr i64 %[[V7]] to i8*
// CHECK: br

// CHECK: %[[V9:.*]] = phi i8* [ null, {{.*}} ], [ %[[V8]], {{.*}} ]
// CHECK: %[[V1:.*]] = ptrtoint i8* %[[V9]] to i64
// CHECK: %[[V11:.*]] = insertvalue { i64, i64 } %[[V2]], i64 %[[V10]], 0
// CHECK: br

// CHECK: %[[V12:.*]] = phi { i64, i64 } [ %[[V2]], {{.*}} ], [ %[[V11]], {{.*}} ]
// CHECK: store { i64, i64 } %[[V12]], { i64, i64 }* %[[METHOD1_ADDR]], align 8
// CHECK: ret void

void testConversion0(MethodTy0 method0, MethodTy1 method1) {
  method1 = method0;
}

// CHECK: define void @_Z15testConversion1M5Base0FvvE(
// CHECK: call i64 @llvm.ptrauth.resign.i64(i64 %{{.*}}, i32 0, i64 [[TYPEDISC0]], i32 0, i64 [[TYPEDISC1]])

void testConversion1(MethodTy0 method0) {
  MethodTy1 method1 = reinterpret_cast<MethodTy1>(method0);
}

// CHECK: define void @_Z15testConversion2M8Derived0FvvE(
// CHECK: call i64 @llvm.ptrauth.resign.i64(i64 %{{.*}}, i32 0, i64 [[TYPEDISC1]], i32 0, i64 [[TYPEDISC0]])

void testConversion2(MethodTy1 method1) {
  MethodTy0 method0 = static_cast<MethodTy0>(method1);
}

// CHECK: define void @_Z15testConversion3M8Derived0FvvE(
// CHECK: call i64 @llvm.ptrauth.resign.i64(i64 %{{.*}}, i32 0, i64 [[TYPEDISC1]], i32 0, i64 [[TYPEDISC0]])

void testConversion3(MethodTy1 method1) {
  MethodTy0 method0 = reinterpret_cast<MethodTy0>(method1);
}

// No need to call @llvm.ptrauth.resign.i64 if the source member function
// pointer is a constant.

// CHECK: define void @_Z15testConversion4v(
// CHECK: %[[METHOD0:.*]] = alloca { i64, i64 }, align 8
// CHECK: store { i64, i64 } { i64 ptrtoint ({ i8*, i32, i64, i64 }* @_ZN5Base08virtual1Ev_vfpthunk_.ptrauth to i64), i64 0 }, { i64, i64 }* %[[METHOD0]], align 8
// CHECK: ret void

void testConversion4() {
  MethodTy0 method0 = reinterpret_cast<MethodTy0>(&Derived0::virtual1);
}

// This code used to crash.
namespace testNonVirtualThunk {
  struct R {};

  struct B0 {
    virtual void bar();
  };

  struct B1 {
    virtual R foo();
  };

  struct D : B0, B1 {
    virtual R foo();
  };

  D d;
}

// CHECK: define void @_Z39test_builtin_ptrauth_type_discriminatorv()
// CHECK: store i32 [[TYPEDISC0]], i32* %
// CHECK: store i32 [[TYPEDISC1]], i32* %
// CHECK: store i32 [[TYPEDISC2]], i32* %

void test_builtin_ptrauth_type_discriminator() {
  unsigned d;
  d = __builtin_ptrauth_type_discriminator(decltype(&Base0::virtual1));
  d = __builtin_ptrauth_type_discriminator(decltype(&Derived0::virtual6));
  d = __builtin_ptrauth_type_discriminator(decltype(&Base1::virtual7));
}

MethodTy1 gmethod0 = reinterpret_cast<MethodTy1>(&Base0::nonvirtual0);
MethodTy0 gmethod1 = reinterpret_cast<MethodTy0>(&Derived0::nonvirtual5);
MethodTy0 gmethod2 = reinterpret_cast<MethodTy0>(&Derived0::virtual1);
