// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -o - %s | FileCheck -check-prefixes=CHECK,NODEBUG,DARWIN %s
// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -debug-info-kind=limited -o - %s | FileCheck -check-prefixes=CHECK,DARWIN %s
// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -stack-protector 1 -o - %s | FileCheck %s -check-prefix=STACK-PROT
// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -stack-protector 2 -o - %s | FileCheck %s -check-prefix=STACK-PROT
// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -stack-protector 3 -o - %s | FileCheck %s -check-prefix=STACK-PROT

// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -o - %s | FileCheck -check-prefixes=CHECK,NODEBUG,ELF %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -debug-info-kind=limited -o - %s | FileCheck -check-prefixes=CHECK,ELF %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -stack-protector 1 -o - %s | FileCheck %s -check-prefix=STACK-PROT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -stack-protector 2 -o - %s | FileCheck %s -check-prefix=STACK-PROT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -emit-llvm -std=c++11 -O1 -disable-llvm-passes -stack-protector 3 -o - %s | FileCheck %s -check-prefix=STACK-PROT


// CHECK: @gmethod0 = global { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base011nonvirtual0Ev, i32 0, i64 [[TYPEDISC1:35591]]) to i64), i64 0 }, align 8
// CHECK: @gmethod1 = global { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN8Derived011nonvirtual5Ev, i32 0, i64 [[TYPEDISC0:22163]]) to i64), i64 0 }, align 8
// CHECK: @gmethod2 = global { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual1Ev_vfpthunk_, i32 0, i64 [[TYPEDISC0]]) to i64), i64 0 }, align 8

// CHECK: @__const._Z13testArrayInitv.p0 = private unnamed_addr constant [1 x { i64, i64 }] [{ i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base011nonvirtual0Ev, i32 0, i64 35591) to i64), i64 0 }], align 8
// CHECK: @__const._Z13testArrayInitv.p1 = private unnamed_addr constant [1 x { i64, i64 }] [{ i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual1Ev_vfpthunk_, i32 0, i64 35591) to i64), i64 0 }], align 8
// CHECK: @__const._Z13testArrayInitv.c0 = private unnamed_addr constant %struct.Class0 { { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base011nonvirtual0Ev, i32 0, i64 35591) to i64), i64 0 } }, align 8
// CHECK: @__const._Z13testArrayInitv.c1 = private unnamed_addr constant %struct.Class0 { { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual1Ev_vfpthunk_, i32 0, i64 35591) to i64), i64 0 } }, align 8

// CHECK: @_ZTV5Base0 = unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI5Base0,
// CHECK-SAME: ptr ptrauth (ptr @_ZN5Base08virtual1Ev, i32 0, i64 55600, ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV5Base0, i32 0, i32 0, i32 2)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN5Base08virtual3Ev, i32 0, i64 53007, ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV5Base0, i32 0, i32 0, i32 3)),
// CHECK-SAME: ptr ptrauth (ptr @_ZN5Base016virtual_variadicEiz, i32 0, i64 7464, ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV5Base0, i32 0, i32 0, i32 4))] }, align 8

typedef __SIZE_TYPE__ size_t;

namespace std {
template <typename _Ep>
class initializer_list {
  const _Ep *__begin_;
  size_t __size_;

  initializer_list(const _Ep *__b, size_t __s);
};
} // namespace std

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

struct Class0 {
  MethodTy1 m0;
};

// CHECK: define{{.*}} void @_ZN5Base08virtual1Ev(

// CHECK: define{{.*}} void @_Z5test0v()
// CHECK: %[[METHOD0:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[VARMETHOD1:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD2:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD3:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD4:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD5:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD6:.*]] = alloca { i64, i64 }, align 8
// CHECK-NEXT: %[[METHOD7:.*]] = alloca { i64, i64 }, align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base011nonvirtual0Ev, i32 0, i64 [[TYPEDISC0]]) to i64), i64 0 }, ptr %[[METHOD0]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual1Ev_vfpthunk_, i32 0, i64 [[TYPEDISC0]]) to i64), i64 0 }, ptr %[[METHOD0]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual3Ev_vfpthunk_, i32 0, i64 [[TYPEDISC0]]) to i64), i64 0 }, ptr %[[METHOD0]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base016virtual_variadicEiz_vfpthunk_, i32 0, i64 34368) to i64), i64 0 }, ptr %[[VARMETHOD1]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base011nonvirtual0Ev, i32 0, i64 [[TYPEDISC1]]) to i64), i64 0 }, ptr %[[METHOD2]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual1Ev_vfpthunk_, i32 0, i64 [[TYPEDISC1]]) to i64), i64 0 }, ptr %[[METHOD2]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual3Ev_vfpthunk_, i32 0, i64 [[TYPEDISC1]]) to i64), i64 0 }, ptr %[[METHOD2]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN8Derived011nonvirtual5Ev, i32 0, i64 [[TYPEDISC1]]) to i64), i64 0 }, ptr %[[METHOD2]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN8Derived08virtual6Ev_vfpthunk_, i32 0, i64 [[TYPEDISC1]]) to i64), i64 0 }, ptr %[[METHOD2]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN8Derived010return_aggEv_vfpthunk_, i32 0, i64 64418) to i64), i64 0 }, ptr %[[METHOD3]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN8Derived04sretEv_vfpthunk_, i32 0, i64 28187) to i64), i64 0 }, ptr %[[METHOD4]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN8Derived011trivial_abiE8TrivialS_vfpthunk_, i32 0, i64 8992) to i64), i64 0 }, ptr %[[METHOD5]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base18virtual7Ev_vfpthunk_, i32 0, i64 [[TYPEDISC2:61596]]) to i64), i64 0 }, ptr %[[METHOD6]], align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN8Derived18virtual7Ev_vfpthunk_, i32 0, i64 25206) to i64), i64 0 }, ptr %[[METHOD7]], align 8
// CHECK-NEXT: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual1Ev_vfpthunk_, i32 0, i64 25206) to i64), i64 0 }, ptr %[[METHOD7]], align 8
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @_ZN5Base08virtual1Ev_vfpthunk_(ptr noundef %[[THIS:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK-NEXT: %[[V0:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK-NEXT: %[[VTABLE:.*]] = load ptr, ptr %[[THIS1]], align 8
// CHECK-NEXT: %[[V2:.*]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK-NEXT: %[[V3:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[V2]], i32 2, i64 0)
// CHECK-NEXT: %[[V4:.*]] = inttoptr i64 %[[V3]] to ptr
// CHECK-NEXT: %[[VFN:.*]] = getelementptr inbounds ptr, ptr %[[V4]], i64 0
// CHECK-NEXT: %[[V5:.*]] = load ptr, ptr %[[VFN]], align 8
// CHECK-NEXT: %[[V6:.*]] = ptrtoint ptr %[[VFN]] to i64
// CHECK-NEXT: %[[V7:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V6]], i64 55600)
// CHECK-NEXT: musttail call void %[[V5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(8) %[[V0]]) [ "ptrauth"(i32 0, i64 %[[V7]]) ]
// CHECK-NEXT: ret void

// CHECK: define linkonce_odr hidden void @_ZN5Base08virtual3Ev_vfpthunk_(ptr noundef %{{.*}})
// CHECK: load ptr, ptr %{{.*}}, align 8
// CHECK: load ptr, ptr %{{.*}}, align 8
// CHECK: %[[VTABLE:.*]] = load ptr, ptr %{{.*}}, align 8
// CHECK: %[[V2:.*]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[V3:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[V2]], i32 2, i64 0)
// CHECK: %[[V4:.*]] = inttoptr i64 %[[V3]] to ptr
// CHECK: getelementptr inbounds ptr, ptr %[[V4]], i64 1
// CHECK: call i64 @llvm.ptrauth.blend(i64 %{{.*}}, i64 53007)

// CHECK: define linkonce_odr hidden void @_ZN5Base016virtual_variadicEiz_vfpthunk_(ptr noundef %[[THIS:.*]], i32 noundef %0, ...)
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK-NEXT: %[[_ADDR:.*]] = alloca i32, align 4
// CHECK-NEXT: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: store i32 %0, ptr %[[_ADDR]], align 4
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK-NEXT: %[[V1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK-NEXT: %[[V2:.*]] = load i32, ptr %[[_ADDR]], align 4
// CHECK-NEXT: %[[VTABLE:.*]] = load ptr, ptr %[[THIS1]], align 8
// CHECK-NEXT: %[[V4:.*]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK-NEXT: %[[V5:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[V4]], i32 2, i64 0)
// CHECK-NEXT: %[[V6:.*]] = inttoptr i64 %[[V5]] to ptr
// CHECK-NEXT: %[[VFN:.*]] = getelementptr inbounds ptr, ptr %[[V6]], i64 2
// CHECK-NEXT: %[[V7:.*]] = load ptr, ptr %[[VFN]], align 8
// CHECK-NEXT: %[[V8:.*]] = ptrtoint ptr %[[VFN]] to i64
// CHECK-NEXT: %[[V9:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V8]], i64 7464)
// CHECK-NEXT: musttail call void (ptr, i32, ...) %[[V7]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(8) %[[V1]], i32 noundef %[[V2]], ...) [ "ptrauth"(i32 0, i64 %[[V9]]) ]
// CHECK-NEXT: ret void

// CHECK: define linkonce_odr hidden void @_ZN8Derived08virtual6Ev_vfpthunk_(ptr noundef %[[THIS:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: %[[VTABLE:.*]] = load ptr, ptr %[[THIS1]], align 8
// CHECK: %[[V1:.*]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[V2:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[V1]], i32 2, i64 0)
// CHECK: %[[V3:.*]] = inttoptr i64 %[[V2]] to ptr
// CHECK: %[[VFN:.*]] = getelementptr inbounds ptr, ptr %[[V3]], i64 3 
// CHECK: %[[V5:.*]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: call i64 @llvm.ptrauth.blend(i64 %[[V5]], i64 55535)

// Check that the return value of the musttail call isn't copied to a temporary.

// CHECK: define linkonce_odr hidden [2 x i64] @_ZN8Derived010return_aggEv_vfpthunk_(ptr noundef %{{.*}})
// CHECK: %[[CALL:.*]] = musttail call [2 x i64] %{{.*}}(ptr noundef nonnull align {{[0-9]+}} dereferenceable(8) %{{.*}}) [ "ptrauth"(i32 0, i64 %{{.*}}) ]
// CHECK-NEXT: ret [2 x i64] %[[CALL]]

// Check that the sret pointer passed to the caller is forwarded to the musttail
// call.

// CHECK: define linkonce_odr hidden void @_ZN8Derived04sretEv_vfpthunk_(ptr dead_on_unwind noalias writable sret(%struct.A1) align 4 %[[AGG_RESULT:.*]], ptr noundef %{{.*}})
// CHECK: musttail call void %{{.*}}(ptr dead_on_unwind writable  sret(%struct.A1) align 4 %[[AGG_RESULT]], ptr noundef nonnull align {{[0-9]+}} dereferenceable(8) %{{.*}}) [ "ptrauth"(i32 0, i64 %{{.*}}) ]
// CHECK-NEXT: ret void

// Check that the thunk function doesn't destruct the trivial_abi argument.

// CHECK: define linkonce_odr hidden void @_ZN8Derived011trivial_abiE8TrivialS_vfpthunk_(ptr noundef %{{.*}}, [2 x i64] %{{.*}})
// NODEBUG-NOT: call
// CHECK: call i64 @llvm.ptrauth.auth(
// NODEBUG-NOT: call
// CHECK: call i64 @llvm.ptrauth.blend(
// NODEBUG-NOT: call
// CHECK: musttail call void
// CHECK-NEXT: ret void

// CHECK: define linkonce_odr hidden void @_ZN5Base18virtual7Ev_vfpthunk_(ptr noundef %[[THIS:.*]])
// CHECK: entry:
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: %[[VTABLE:.*]] = load ptr, ptr %[[THIS1]], align 8
// CHECK: %[[V1:.*]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[V2:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[V1]], i32 2, i64 0)
// CHECK: %[[V3:.*]] = inttoptr i64 %[[V2]] to ptr
// CHECK: getelementptr inbounds ptr, ptr %[[V3]], i64 0

// CHECK: define linkonce_odr hidden void @_ZN8Derived18virtual7Ev_vfpthunk_(ptr noundef %[[THIS:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: %[[VTABLE:.*]] = load ptr, ptr %[[THIS1]], align 8
// CHECK: %[[V1:.*]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[V2:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[V1]], i32 2, i64 0)
// CHECK: %[[V3:.*]] = inttoptr i64 %[[V2]] to ptr
// CHECK: getelementptr inbounds ptr, ptr %[[V3]], i64 3

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

// CHECK: define{{.*}} void @_Z5test1P5Base0MS_FvvE(ptr noundef %[[A0:.*]], [2 x i64] %[[A1_COERCE:.*]])
// CHECK: %[[A1:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[A0_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[A1_ADDR:.*]] = alloca { i64, i64 }, align 8
// CHECK: store [2 x i64] %[[A1_COERCE]], ptr %[[A1]], align 8
// CHECK: %[[A11:.*]] = load { i64, i64 }, ptr %[[A1]], align 8
// CHECK: store ptr %[[A0]], ptr %[[A0_ADDR]], align 8
// CHECK: store { i64, i64 } %[[A11]], ptr %[[A1_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[A0_ADDR]], align 8
// CHECK: %[[V2:.*]] = load { i64, i64 }, ptr %[[A1_ADDR]], align 8
// CHECK: %[[MEMPTR_ADJ:.*]] = extractvalue { i64, i64 } %[[V2]], 1
// CHECK: %[[MEMPTR_ADJ_SHIFTED:.*]] = ashr i64 %[[MEMPTR_ADJ]], 1
// CHECK: %[[V4:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 %[[MEMPTR_ADJ_SHIFTED]]
// CHECK: %[[MEMPTR_PTR:.*]] = extractvalue { i64, i64 } %[[V2]], 0
// CHECK: %[[V5:.*]] = and i64 %[[MEMPTR_ADJ]], 1
// CHECK: %[[MEMPTR_ISVIRTUAL:.*]] = icmp ne i64 %[[V5]], 0
// CHECK: br i1 %[[MEMPTR_ISVIRTUAL]]

// CHECK:  %[[VTABLE:.*]] = load ptr, ptr %[[V4]], align 8
// CHECK:  %[[V7:.*]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK:  %[[V8:.*]] = call i64 @llvm.ptrauth.auth(i64 %[[V7]], i32 2, i64 0)
// CHECK:  %[[V9:.*]] = inttoptr i64 %[[V8]] to ptr
// DARWIN: %[[V10:.*]] = trunc i64 %[[MEMPTR_PTR]] to i32
// DARWIN: %[[V11:.*]] = zext i32 %[[V10]] to i64
// DARWIN: %[[V12:.*]] = getelementptr i8, ptr %[[V9]], i64 %[[V11]]
// ELF:    %[[V12:.*]] = getelementptr i8, ptr %[[V9]], i64 %[[MEMPTR_PTR]]
// CHECK:  %[[MEMPTR_VIRTUALFN:.*]] = load ptr, ptr %[[V12]], align 8
// CHECK:  br

// CHECK: %[[MEMPTR_NONVIRTUALFN:.*]] = inttoptr i64 %[[MEMPTR_PTR]] to ptr
// CHECK: br

// CHECK: %[[V14:.*]] = phi ptr [ %[[MEMPTR_VIRTUALFN]], {{.*}} ], [ %[[MEMPTR_NONVIRTUALFN]], {{.*}} ]
// CHECK: %[[V15:.*]] = phi i64 [ 0, {{.*}} ], [ [[TYPEDISC0]], {{.*}} ]
// CHECK: call void %[[V14]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(8) %[[V4]]) [ "ptrauth"(i32 0, i64 %[[V15]]) ]
// CHECK: ret void

void test1(Base0 *a0, MethodTy0 a1) {
  (a0->*a1)();
}

// CHECK: define{{.*}} void @_Z15testConversion0M5Base0FvvEM8Derived0FvvE([2 x i64] %[[METHOD0_COERCE:.*]], [2 x i64] %[[METHOD1_COERCE:.*]])
// CHECK: %[[METHOD0:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[METHOD1:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[METHOD0_ADDR:.*]] = alloca { i64, i64 }, align 8
// CHECK: %[[METHOD1_ADDR:.*]] = alloca { i64, i64 }, align 8
// CHECK: store [2 x i64] %[[METHOD0_COERCE]], ptr %[[METHOD0]], align 8
// CHECK: %[[METHOD01:.*]] = load { i64, i64 }, ptr %[[METHOD0]], align 8
// CHECK: store [2 x i64] %[[METHOD1_COERCE]], ptr %[[METHOD1]], align 8
// CHECK: %[[METHOD12:.*]] = load { i64, i64 }, ptr %[[METHOD1]], align 8
// CHECK: store { i64, i64 } %[[METHOD01]], ptr %[[METHOD0_ADDR]], align 8
// CHECK: store { i64, i64 } %[[METHOD12]], ptr %[[METHOD1_ADDR]], align 8
// CHECK: %[[V2:.*]] = load { i64, i64 }, ptr %[[METHOD0_ADDR]], align 8
// CHECK: %[[MEMPTR_PTR:.*]] = extractvalue { i64, i64 } %[[V2]], 0
// CHECK: %[[MEMPTR_ADJ:.*]] = extractvalue { i64, i64 } %[[V2]], 1
// CHECK: %[[V3:.*]] = and i64 %[[MEMPTR_ADJ]], 1
// CHECK: %[[IS_VIRTUAL_OFFSET:.*]] = icmp ne i64 %[[V3]], 0
// CHECK: br i1 %[[IS_VIRTUAL_OFFSET]]

// CHECK: %[[V4:.*]] = inttoptr i64 %[[MEMPTR_PTR]] to ptr
// CHECK: %[[V5:.*]] = icmp ne ptr %[[V4]], null
// CHECK: br i1 %[[V5]]

// CHECK: %[[V6:.*]] = ptrtoint ptr %[[V4]] to i64
// CHECK: %[[V7:.*]] = call i64 @llvm.ptrauth.resign(i64 %[[V6]], i32 0, i64 [[TYPEDISC0]], i32 0, i64 [[TYPEDISC1]])
// CHECK: %[[V8:.*]] = inttoptr i64 %[[V7]] to ptr
// CHECK: br

// CHECK: %[[V9:.*]] = phi ptr [ null, {{.*}} ], [ %[[V8]], {{.*}} ]
// CHECK: %[[V1:.*]] = ptrtoint ptr %[[V9]] to i64
// CHECK: %[[V11:.*]] = insertvalue { i64, i64 } %[[V2]], i64 %[[V1]], 0
// CHECK: br

// CHECK: %[[V12:.*]] = phi { i64, i64 } [ %[[V2]], {{.*}} ], [ %[[V11]], {{.*}} ]
// CHECK: store { i64, i64 } %[[V12]], ptr %[[METHOD1_ADDR]], align 8
// CHECK: ret void

void testConversion0(MethodTy0 method0, MethodTy1 method1) {
  method1 = method0;
}

// CHECK: define{{.*}} void @_Z15testConversion1M5Base0FvvE(
// CHECK: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 [[TYPEDISC0]], i32 0, i64 [[TYPEDISC1]])

void testConversion1(MethodTy0 method0) {
  MethodTy1 method1 = reinterpret_cast<MethodTy1>(method0);
}

// CHECK: define{{.*}} void @_Z15testConversion2M8Derived0FvvE(
// CHECK: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 [[TYPEDISC1]], i32 0, i64 [[TYPEDISC0]])

void testConversion2(MethodTy1 method1) {
  MethodTy0 method0 = static_cast<MethodTy0>(method1);
}

// CHECK: define{{.*}} void @_Z15testConversion3M8Derived0FvvE(
// CHECK: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 [[TYPEDISC1]], i32 0, i64 [[TYPEDISC0]])

void testConversion3(MethodTy1 method1) {
  MethodTy0 method0 = reinterpret_cast<MethodTy0>(method1);
}

// No need to call @llvm.ptrauth.resign if the source member function
// pointer is a constant.

// CHECK: define{{.*}} void @_Z15testConversion4v(
// CHECK: %[[METHOD0:.*]] = alloca { i64, i64 }, align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual1Ev_vfpthunk_, i32 0, i64 [[TYPEDISC0]]) to i64), i64 0 }, ptr %[[METHOD0]], align 8
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

// CHECK: define internal void @_ZN22TestAnonymousNamespace12_GLOBAL__N_11S3fooEv_vfpthunk_(

namespace TestAnonymousNamespace {
namespace {
struct S {
  virtual void foo(){};
};
} // namespace

void test() {
  auto t = &S::foo;
}
} // namespace TestAnonymousNamespace

MethodTy1 gmethod0 = reinterpret_cast<MethodTy1>(&Base0::nonvirtual0);
MethodTy0 gmethod1 = reinterpret_cast<MethodTy0>(&Derived0::nonvirtual5);
MethodTy0 gmethod2 = reinterpret_cast<MethodTy0>(&Derived0::virtual1);

// CHECK-LABEL: define{{.*}} void @_Z13testArrayInitv()
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %p0, ptr align 8 @__const._Z13testArrayInitv.p0, i64 16, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %p1, ptr align 8 @__const._Z13testArrayInitv.p1, i64 16, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %c0, ptr align 8 @__const._Z13testArrayInitv.c0, i64 16, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %c1, ptr align 8 @__const._Z13testArrayInitv.c1, i64 16, i1 false)
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base011nonvirtual0Ev, i32 0, i64 [[TYPEDISC1]]) to i64), i64 0 }, ptr %{{.*}} align 8
// CHECK: store { i64, i64 } { i64 ptrtoint (ptr ptrauth (ptr @_ZN5Base08virtual1Ev_vfpthunk_, i32 0, i64 [[TYPEDISC1]]) to i64), i64 0 }, ptr %{{.*}}, align 8

void initList(std::initializer_list<MethodTy1>);

void testArrayInit() {
  MethodTy1 p0[] = {&Base0::nonvirtual0};
  MethodTy1 p1[] = {&Base0::virtual1};
  Class0 c0{&Base0::nonvirtual0};
  Class0 c1{&Base0::virtual1};
  initList({&Base0::nonvirtual0});
  initList({&Base0::virtual1});
}



// STACK-PROT: define {{.*}}_vfpthunk{{.*}}[[ATTRS:#[0-9]+]]
// STACK-PROT: attributes [[ATTRS]] =
// STACK-PROT-NOT: ssp
// STACK-PROT-NOT: sspstrong
// STACK-PROT-NOT: sspreq
// STACK-PROT-NEXT: attributes

// CHECK: define{{.*}} void @_Z15testConvertNullv(
// CHECK: %[[T:.*]] = alloca { i64, i64 },
// store { i64, i64 } zeroinitializer, { i64, i64 }* %[[T]],

void testConvertNull() {
  VariadicMethodTy0 t = (VariadicMethodTy0)(MethodTy0{});
}
