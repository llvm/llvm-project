// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck --check-prefixes=CHECK,LINUX,LINUX_AIX %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -emit-llvm %s -O2 -disable-llvm-passes -o - -triple x86_64-linux-gnu | FileCheck --check-prefixes=CHECK,LINUX,LINUX_AIX,CHECK-OPT %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -femulated-tls -emit-llvm %s -o - \
// RUN:     -triple x86_64-linux-gnu 2>&1 | FileCheck --check-prefixes=CHECK,LINUX,LINUX_AIX %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -emit-llvm %s -o - -triple x86_64-apple-darwin12 | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -emit-llvm %s -o - -triple powerpc64-unknown-aix-xcoff | FileCheck --check-prefixes=CHECK,AIX,LINUX_AIX %s

// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -fno-use-cxa-atexit -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck --check-prefixes=CHECK,LINUX,LINUX_AIX %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -fno-use-cxa-atexit -emit-llvm %s -O2 -disable-llvm-passes -o - -triple x86_64-linux-gnu | FileCheck --check-prefixes=CHECK,LINUX,LINUX_AIX,CHECK-OPT %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -fno-use-cxa-atexit -femulated-tls -emit-llvm %s -o - \
// RUN:     -triple x86_64-linux-gnu 2>&1 | FileCheck --check-prefixes=CHECK,LINUX,LINUX_AIX %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -fno-use-cxa-atexit -emit-llvm %s -o - -triple x86_64-apple-darwin12 | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s

int f();
int g();

// LINUX_AIX-DAG: @a ={{.*}} thread_local global i32 0
// DARWIN-DAG: @a = internal thread_local global i32 0
thread_local int a = f();
extern thread_local int b;
// CHECK-DAG: @c ={{.*}} global i32 0
int c = b;
// CHECK-DAG: @_ZL1d = internal thread_local global i32 0
static thread_local int d = g();

struct U { static thread_local int m; };
// LINUX_AIX-DAG: @_ZN1U1mE ={{.*}} thread_local global i32 0
// DARWIN-DAG: @_ZN1U1mE = internal thread_local global i32 0
thread_local int U::m = f();

namespace MismatchedInitType {
  // Check that we don't crash here when we're forced to create a new global
  // variable (with a different type) when we add the initializer.
  union U {
    int a;
    float f;
    constexpr U() : f(0.0) {}
  };
  static thread_local U u;
  void *p = &u;
}

template<typename T> struct V { static thread_local int m; };
template<typename T> thread_local int V<T>::m = g();

template<typename T> struct W { static thread_local int m; };
template<typename T> thread_local int W<T>::m = 123;

struct Dtor { ~Dtor(); };
template<typename T> struct X { static thread_local Dtor m; };
template<typename T> thread_local Dtor X<T>::m;

// CHECK-DAG: @e ={{.*}} global
void *e = V<int>::m + W<int>::m + &X<int>::m;

template thread_local int V<float>::m;
template thread_local int W<float>::m;
template thread_local Dtor X<float>::m;

extern template thread_local int V<char>::m;
extern template thread_local int W<char>::m;
extern template thread_local Dtor X<char>::m;

void *e2 = V<char>::m + W<char>::m + &X<char>::m;

// CHECK-DAG: @_ZN1VIiE1mE = linkonce_odr thread_local global i32 0
// CHECK-DAG: @_ZN1WIiE1mE = linkonce_odr thread_local global i32 123
// CHECK-DAG: @_ZN1XIiE1mE = linkonce_odr thread_local global {{.*}}
// CHECK-DAG: @_ZN1VIfE1mE = weak_odr thread_local global i32 0
// CHECK-DAG: @_ZN1WIfE1mE = weak_odr thread_local global i32 123
// CHECK-DAG: @_ZN1XIfE1mE = weak_odr thread_local global {{.*}}

// CHECK-DAG: @_ZZ1fvE1n = internal thread_local global i32 0

// CHECK-DAG: @_ZGVZ1fvE1n = internal thread_local global i8 0

// CHECK-DAG: @_ZZ8tls_dtorvE1s = internal thread_local global
// CHECK-DAG: @_ZGVZ8tls_dtorvE1s = internal thread_local global i8 0

// CHECK-DAG: @_ZZ8tls_dtorvE1t = internal thread_local global
// CHECK-DAG: @_ZGVZ8tls_dtorvE1t = internal thread_local global i8 0

// CHECK-DAG: @_ZZ8tls_dtorvE1u = internal thread_local global
// CHECK-DAG: @_ZGVZ8tls_dtorvE1u = internal thread_local global i8 0
// CHECK-DAG: @_ZGRZ8tls_dtorvE1u_ = internal thread_local global

// CHECK-DAG: @_ZGVN1VIiE1mE = linkonce_odr thread_local global i64 0

// CHECK-DAG: @__tls_guard = internal thread_local global i8 0

// CHECK-DAG: @llvm.global_ctors = appending global {{.*}} @[[GLOBAL_INIT:[^ ]*]]

// LINUX_AIX-DAG: @_ZTH1a ={{.*}} alias void (), void ()* @__tls_init
// DARWIN-DAG: @_ZTH1a = internal alias void (), void ()* @__tls_init
// LINUX_AIX-DAG: @_ZTHN1U1mE ={{.*}} alias void (), void ()* @__tls_init
// DARWIN-DAG: @_ZTHN1U1mE = internal alias void (), void ()* @__tls_init
// CHECK-DAG: @_ZTHN1VIiE1mE = linkonce_odr alias void (), void ()* @[[V_M_INIT:[^, ]*]]
// CHECK-DAG: @_ZTHN1XIiE1mE = linkonce_odr alias void (), void ()* @[[X_M_INIT:[^, ]*]]
// CHECK-DAG: @_ZTHN1VIfE1mE = weak_odr alias void (), void ()* @[[VF_M_INIT:[^, ]*]]
// CHECK-DAG: @_ZTHN1XIfE1mE = weak_odr alias void (), void ()* @[[XF_M_INIT:[^, ]*]]
// FIXME: We really want a CHECK-DAG-NOT for these.
// CHECK-NOT: @_ZTHN1WIiE1mE =
// CHECK-NOT: @_ZTHN1WIfE1mE =
// CHECK-NOT: @_ZTHL1d =


// Individual variable initialization functions:

// CHECK: define {{.*}} @[[A_INIT:.*]]()
// CHECK: call{{.*}} i32 @_Z1fv()
// CHECK: [[A_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @a)
// CHECK-NEXT: store i32 {{.*}}, i32* [[A_ADDR]], align 4

// CHECK-LABEL: define{{.*}} i32 @_Z1fv()
int f() {
  // CHECK: %[[GUARD:.*]] = load i8, i8* @_ZGVZ1fvE1n, align 1
  // CHECK: %[[NEED_INIT:.*]] = icmp eq i8 %[[GUARD]], 0
  // CHECK: br i1 %[[NEED_INIT]]{{.*}}

  // CHECK: %[[CALL:.*]] = call{{.*}} i32 @_Z1gv()
  // CHECK: [[N_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZZ1fvE1n)
  // CHECK: store i32 %[[CALL]], i32* [[N_ADDR]], align 4
  // CHECK: store i8 1, i8* @_ZGVZ1fvE1n
  // CHECK: br label
  static thread_local int n = g();

  // CHECK: [[N_ADDR2:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZZ1fvE1n)
  // CHECK: load i32, i32* [[N_ADDR2]], align 4
  return n;
}

// CHECK: define {{.*}} @[[C_INIT:.*]]()
// LINUX_AIX: call i32* @_ZTW1b()
// DARWIN: call cxx_fast_tlscc i32* @_ZTW1b()
// CHECK-NEXT: load i32, i32* %{{.*}}, align 4
// CHECK-NEXT: store i32 %{{.*}}, i32* @c, align 4

// LINUX_AIX-LABEL: define linkonce_odr hidden noundef i32* @_ZTW1b()
// LINUX: br i1 icmp ne (void ()* @_ZTH1b, void ()* null),
// AIX-NOT: br i1 icmp ne (void ()* @_ZTH1b, void ()* null),
// not null:
// LINUX_AIX: call void @_ZTH1b()
// LINUX: br label
// AIX-NOT: br label
// finally:
// LINUX_AIX: [[B_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @b)
// LINUX_AIX: ret i32* [[B_ADDR]]
// DARWIN-LABEL: declare cxx_fast_tlscc noundef i32* @_ZTW1b()
// There is no definition of the thread wrapper on Darwin for external TLV.

// CHECK: define {{.*}} @[[D_INIT:.*]]()
// CHECK: call{{.*}} i32 @_Z1gv()
// CHECK-NEXT: [[D_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZL1d)
// CHECK-NEXT: store i32 %{{.*}}, i32* [[D_ADDR]], align 4

// CHECK: define {{.*}} @[[U_M_INIT:.*]]()
// CHECK: call{{.*}} i32 @_Z1fv()
// CHECK-NEXT: [[UM_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZN1U1mE)
// CHECK-NEXT: store i32 %{{.*}}, i32* [[UM_ADDR]], align 4

// CHECK: define {{.*}} @[[E_INIT:.*]]()
// LINUX_AIX: call i32* @_ZTWN1VIiE1mE()
// DARWIN: call cxx_fast_tlscc i32* @_ZTWN1VIiE1mE()
// CHECK-NEXT: load i32, i32* %{{.*}}, align 4
// LINUX_AIX: call {{.*}}* @_ZTWN1XIiE1mE()
// DARWIN: call cxx_fast_tlscc {{.*}}* @_ZTWN1XIiE1mE()
// CHECK: store {{.*}} @e

// LINUX_AIX-LABEL: define weak_odr hidden noundef i32* @_ZTWN1VIiE1mE()
// DARWIN-LABEL: define weak_odr hidden cxx_fast_tlscc noundef i32* @_ZTWN1VIiE1mE()
// LINUX_AIX: call void @_ZTHN1VIiE1mE()
// DARWIN: call cxx_fast_tlscc void @_ZTHN1VIiE1mE()
// CHECK: [[VM_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZN1VIiE1mE)
// CHECK: ret i32* [[VM_ADDR]]

// LINUX_AIX-LABEL: define weak_odr hidden noundef i32* @_ZTWN1WIiE1mE()
// DARWIN-LABEL: define weak_odr hidden cxx_fast_tlscc noundef i32* @_ZTWN1WIiE1mE()
// CHECK-NOT: call
// CHECK: [[WM_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZN1WIiE1mE)
// CHECK: ret i32* [[WM_ADDR]]

// LINUX_AIX-LABEL: define weak_odr hidden {{.*}}* @_ZTWN1XIiE1mE()
// DARWIN-LABEL: define weak_odr hidden cxx_fast_tlscc {{.*}}* @_ZTWN1XIiE1mE()
// LINUX_AIX: call void @_ZTHN1XIiE1mE()
// DARWIN: call cxx_fast_tlscc void @_ZTHN1XIiE1mE()
// CHECK: [[XM_ADDR:%.+]] = call align 1 %struct.Dtor* @llvm.threadlocal.address.p0s_struct.Dtors(%struct.Dtor* align 1 @_ZN1XIiE1mE)
// CHECK: ret {{.*}}* [[XM_ADDR]]

// LINUX_AIX: define internal void @[[VF_M_INIT]]()
// DARWIN: define internal cxx_fast_tlscc void @[[VF_M_INIT]]()
// CHECK-NOT: comdat
// CHECK: load i8, i8* bitcast (i64* @_ZGVN1VIfE1mE to i8*)
// CHECK: %[[VF_M_INITIALIZED:.*]] = icmp eq i8 %{{.*}}, 0
// CHECK: br i1 %[[VF_M_INITIALIZED]],
// need init:
// CHECK: store i8 1, i8* bitcast (i64* @_ZGVN1VIfE1mE to i8*)
// CHECK: call{{.*}} i32 @_Z1gv()
// CHECK: [[VFM_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZN1VIfE1mE)
// CHECK: store i32 %{{.*}}, i32* [[VFM_ADDR]], align 4
// CHECK: br label

// LINUX_AIX: define internal void @[[XF_M_INIT]]()
// DARWIN: define internal cxx_fast_tlscc void @[[XF_M_INIT]]()
// CHECK-NOT: comdat
// CHECK: load i8, i8* bitcast (i64* @_ZGVN1XIfE1mE to i8*)
// CHECK: %[[XF_M_INITIALIZED:.*]] = icmp eq i8 %{{.*}}, 0
// CHECK: br i1 %[[XF_M_INITIALIZED]],
// need init:
// CHECK: store i8 1, i8* bitcast (i64* @_ZGVN1XIfE1mE to i8*)
// AIX-NOT: br
// LINUX: call {{.*}}__cxa_thread_atexit
// AIX: call {{.*}}__pt_atexit_np
// DARWIN: call {{.*}}_tlv_atexit
// CHECK: br label

// LINUX: declare i32 @__cxa_thread_atexit(void (i8*)*, i8*, i8*)
// AIX: declare i32 @__pt_atexit_np(i32, i32 (i32, ...)*, ...)
// DARWIN: declare i32 @_tlv_atexit(void (i8*)*, i8*, i8*)

// DARWIN: declare cxx_fast_tlscc noundef i32* @_ZTWN1VIcE1mE()
// LINUX_AIX: define linkonce_odr hidden noundef i32* @_ZTWN1VIcE1mE() {{#[0-9]+}}{{( comdat)?}} {
// LINUX: br i1 icmp ne (void ()* @_ZTHN1VIcE1mE,
// AIX-NOT: br i1 icmp ne (void ()* @_ZTHN1VIcE1mE
// LINUX_AIX: call void @_ZTHN1VIcE1mE()
// LINUX_AIX: [[VEM_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZN1VIcE1mE)
// LINUX_AIX: ret i32* [[VEM_ADDR]]

// DARWIN: declare cxx_fast_tlscc noundef i32* @_ZTWN1WIcE1mE()
// LINUX_AIX: define linkonce_odr hidden noundef i32* @_ZTWN1WIcE1mE() {{#[0-9]+}}{{( comdat)?}} {
// LINUX: br i1 icmp ne (void ()* @_ZTHN1WIcE1mE,
// AIX-NOT: br i1 icmp ne (void ()* @_ZTHN1WIcE1mE,
// LINUX_AIX: call void @_ZTHN1WIcE1mE()
// LINUX_AIX: [[WEM_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZN1WIcE1mE)
// LINUX_AIX: ret i32* [[WEM_ADDR]]

// DARWIN: declare cxx_fast_tlscc {{.*}}* @_ZTWN1XIcE1mE()
// LINUX_AIX: define linkonce_odr hidden {{.*}}* @_ZTWN1XIcE1mE() {{#[0-9]+}}{{( comdat)?}} {
// LINUX: br i1 icmp ne (void ()* @_ZTHN1XIcE1mE,
// AIX-NOT: br i1 icmp ne (void ()* @_ZTHN1XIcE1mE,
// LINUX_AIX: call void @_ZTHN1XIcE1mE()
// LINUX_AIX: [[XEM_ADDR:%.+]] = call align 1 %struct.Dtor* @llvm.threadlocal.address.p0s_struct.Dtors(%struct.Dtor* align 1 @_ZN1XIcE1mE)
// LINUX_AIX: ret {{.*}}* [[XEM_ADDR]]

struct S { S(); ~S(); };
struct T { ~T(); };

// CHECK-LABEL: define{{.*}} void @_Z8tls_dtorv()
void tls_dtor() {
  // CHECK: load i8, i8* @_ZGVZ8tls_dtorvE1s
  // CHECK: call void @_ZN1SC1Ev(%struct.S* {{[^,]*}} @_ZZ8tls_dtorvE1s)
  // LINUX: call i32 @__cxa_thread_atexit({{.*}}@_ZN1SD1Ev {{.*}} @_ZZ8tls_dtorvE1s{{.*}} @__dso_handle
  // AIX: call i32 (i32, i32 (i32, ...)*, ...) @__pt_atexit_np(i32 0, {{.*}}@__dtor__ZZ8tls_dtorvE1s){{.*}}
  // DARWIN: call i32 @_tlv_atexit({{.*}}@_ZN1SD1Ev {{.*}} @_ZZ8tls_dtorvE1s{{.*}} @__dso_handle
  // CHECK: store i8 1, i8* @_ZGVZ8tls_dtorvE1s
  static thread_local S s;

  // CHECK: load i8, i8* @_ZGVZ8tls_dtorvE1t
  // CHECK-NOT: _ZN1T
  // LINUX: call i32 @__cxa_thread_atexit({{.*}}@_ZN1TD1Ev {{.*}}@_ZZ8tls_dtorvE1t{{.*}} @__dso_handle
  // AIX: call i32 (i32, i32 (i32, ...)*, ...) @__pt_atexit_np(i32 0, {{.*}}@__dtor__ZZ8tls_dtorvE1t){{.*}}
  // DARWIN: call i32 @_tlv_atexit({{.*}}@_ZN1TD1Ev {{.*}}@_ZZ8tls_dtorvE1t{{.*}} @__dso_handle
  // CHECK: store i8 1, i8* @_ZGVZ8tls_dtorvE1t
  static thread_local T t;

  // CHECK: load i8, i8* @_ZGVZ8tls_dtorvE1u
  // CHECK: call void @_ZN1SC1Ev(%struct.S* {{[^,]*}} @_ZGRZ8tls_dtorvE1u_)
  // LINUX: call i32 @__cxa_thread_atexit({{.*}}@_ZN1SD1Ev {{.*}} @_ZGRZ8tls_dtorvE1u_{{.*}} @__dso_handle
  // AIX: call i32 (i32, i32 (i32, ...)*, ...) @__pt_atexit_np(i32 0, {{.*}}__dtor__ZZ8tls_dtorvE1u){{.*}}
  // DARWIN: call i32 @_tlv_atexit({{.*}}@_ZN1SD1Ev {{.*}} @_ZGRZ8tls_dtorvE1u_{{.*}} @__dso_handle
  // CHECK: store i8 1, i8* @_ZGVZ8tls_dtorvE1u
  static thread_local const S &u = S();
}

// AIX: define {{.*}}@__dtor__ZZ8tls_dtorvE1s(i32 noundef signext %0, ...){{.*}}{
// AIX: entry:
// AIX:   %.addr = alloca i32, align 4
// AIX:   store i32 %0, i32* %.addr, align 4
// AIX:   call void @_ZN1SD1Ev(%struct.S* @_ZZ8tls_dtorvE1s)
// AIX:   ret i32 0
// AIX: }

// AIX: define {{.*}}@__dtor__ZZ8tls_dtorvE1t(i32 noundef signext %0, ...){{.*}}{
// AIX: entry:
// AIX:   %.addr = alloca i32, align 4
// AIX:   store i32 %0, i32* %.addr, align 4
// AIX:   call void @_ZN1TD1Ev(%struct.T* @_ZZ8tls_dtorvE1t)
// AIX:   ret i32 0
// AIX: }

// AIX: define {{.*}}@__dtor__ZZ8tls_dtorvE1u(i32 noundef signext %0, ...){{.*}}{
// AIX: entry:
// AIX:   %.addr = alloca i32, align 4
// AIX:   store i32 %0, i32* %.addr, align 4
// AIX:   call void @_ZN1SD1Ev(%struct.S* @_ZGRZ8tls_dtorvE1u_)
// AIX:   ret i32 0
// AIX: }

// CHECK: define {{.*}} @_Z7PR15991v(
int PR15991() {
  thread_local int n;
  auto l = [] { return n; };
  return l();
}

struct PR19254 {
  static thread_local int n;
  int f();
};
// CHECK: define {{.*}} @_ZN7PR192541fEv(
int PR19254::f() {
  // LINUX_AIX: call void @_ZTHN7PR192541nE(
  // DARWIN: call cxx_fast_tlscc i32* @_ZTWN7PR192541nE(
  return this->n;
}

namespace {
thread_local int anon_i{f()};
}
void set_anon_i() {
  anon_i = 2;
}
// LINUX_AIX-LABEL: define internal noundef i32* @_ZTWN12_GLOBAL__N_16anon_iE()
// DARWIN-LABEL: define internal cxx_fast_tlscc noundef i32* @_ZTWN12_GLOBAL__N_16anon_iE()

// LINUX_AIX: define internal void @[[V_M_INIT]]()
// DARWIN: define internal cxx_fast_tlscc void @[[V_M_INIT]]()
// CHECK-NOT: comdat
// CHECK: load i8, i8* bitcast (i64* @_ZGVN1VIiE1mE to i8*)
// CHECK: %[[V_M_INITIALIZED:.*]] = icmp eq i8 %{{.*}}, 0
// CHECK: br i1 %[[V_M_INITIALIZED]],
// need init:
// CHECK: store i8 1, i8* bitcast (i64* @_ZGVN1VIiE1mE to i8*)
// CHECK: call{{.*}} i32 @_Z1gv()
// CHECK: [[VEM_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZN1VIiE1mE)
// CHECK: store i32 %{{.*}}, i32* [[VEM_ADDR]], align 4
// CHECK: br label

// LINUX_AIX: define internal void @[[X_M_INIT]]()
// DARWIN: define internal cxx_fast_tlscc void @[[X_M_INIT]]()
// CHECK-NOT: comdat
// CHECK: load i8, i8* bitcast (i64* @_ZGVN1XIiE1mE to i8*)
// CHECK: %[[X_M_INITIALIZED:.*]] = icmp eq i8 %{{.*}}, 0
// CHECK: br i1 %[[X_M_INITIALIZED]],
// need init:
// CHECK: store i8 1, i8* bitcast (i64* @_ZGVN1XIiE1mE to i8*)
// LINUX: call {{.*}}__cxa_thread_atexit
// AIX: call {{.*}}__pt_atexit_np
// DARWIN: call {{.*}}_tlv_atexit
// CHECK: br label

// CHECK: define {{.*}}@[[GLOBAL_INIT:.*]]()
// CHECK: call void @[[C_INIT]]()
// CHECK: call void @[[E_INIT]]()


// CHECK: define {{.*}}@__tls_init()
// CHECK: load i8, i8* @__tls_guard
// CHECK: %[[NEED_TLS_INIT:.*]] = icmp eq i8 %{{.*}}, 0
// CHECK: br i1 %[[NEED_TLS_INIT]],
// init:
// CHECK: store i8 1, i8* @__tls_guard
// CHECK-OPT: call {}* @llvm.invariant.start.p0i8(i64 1, i8* @__tls_guard)
// CHECK-NOT: call void @[[V_M_INIT]]()
// CHECK: call void @[[A_INIT]]()
// CHECK-NOT: call void @[[V_M_INIT]]()
// CHECK: call void @[[D_INIT]]()
// CHECK-NOT: call void @[[V_M_INIT]]()
// CHECK: call void @[[U_M_INIT]]()
// CHECK-NOT: call void @[[V_M_INIT]]()


// LINUX_AIX: define weak_odr hidden noundef i32* @_ZTW1a()
// DARWIN: define cxx_fast_tlscc noundef i32* @_ZTW1a()
// LINUX_AIX:   call void @_ZTH1a()
// DARWIN: call cxx_fast_tlscc void @_ZTH1a()
// CHECK:   [[A_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @a)
// CHECK:   ret i32* [[A_ADDR]]
// CHECK: }


// Should not emit a thread wrapper for internal-linkage unused variable 'd'.
// We separately check that 'd' does in fact get initialized with the other
// thread-local variables in this TU.
// CHECK-NOT: define {{.*}} @_ZTWL1d()

// LINUX_AIX-LABEL: define weak_odr hidden noundef i32* @_ZTWN1U1mE()
// DARWIN-LABEL: define cxx_fast_tlscc noundef i32* @_ZTWN1U1mE()
// LINUX_AIX: call void @_ZTHN1U1mE()
// DARWIN: call cxx_fast_tlscc void @_ZTHN1U1mE()
// CHECK: [[UM_ADDR:%.+]] = call align 4 i32* @llvm.threadlocal.address.p0i32(i32* align 4 @_ZN1U1mE)
// CHECK: ret i32* [[UM_ADDR]]

// LINUX_AIX: declare extern_weak void @_ZTH1b() [[ATTR:#[0-9]+]]

// AIX: define linkonce_odr void @_ZTHN1WIiE1mE(){{.*}} {
// AIX-NEXT: ret void
// AIX-NEXT: }
// CHECK-NOT: @_ZTHN1WIfE1mE =
// AIX: define weak_odr void @_ZTHN1WIfE1mE(){{.*}} {
// AIX-NEXT: ret void
// AIX-NEXT: }

// LINUX_AIX: attributes [[ATTR]] = { {{.+}} }
