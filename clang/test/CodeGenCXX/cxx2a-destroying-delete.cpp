// RUN: %clang_cc1 -no-opaque-pointers -std=c++2a -fexceptions -emit-llvm %s -triple x86_64-linux-gnu -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ITANIUM,CHECK-64BIT
// RUN: %clang_cc1 -no-opaque-pointers -std=c++2a -fexceptions -emit-llvm %s -triple x86_64-windows -o - | FileCheck %s --check-prefixes=CHECK,CHECK-MSABI,CHECK-MSABI64,CHECK-64BIT
// RUN: %clang_cc1 -no-opaque-pointers -std=c++2a -fexceptions -emit-llvm %s -triple i386-windows -o - | FileCheck %s --check-prefixes=CHECK,CHECK-MSABI,CHECK-MSABI32,CHECK-32BIT

// PR46908: ensure the IR passes the verifier with optimizations enabled.
// RUN: %clang_cc1 -no-opaque-pointers -std=c++2a -fexceptions -emit-llvm-only %s -triple x86_64-linux-gnu -O2
// RUN: %clang_cc1 -no-opaque-pointers -std=c++2a -fexceptions -emit-llvm-only %s -triple x86_64-windows -O2
// RUN: %clang_cc1 -no-opaque-pointers -std=c++2a -fexceptions -emit-llvm-only %s -triple i386-windows -O2

namespace std {
  using size_t = decltype(sizeof(0));
  enum class align_val_t : size_t;
  struct destroying_delete_t {};
}

struct A {
  void *data;
  ~A();
  void operator delete(A*, std::destroying_delete_t);
};
void delete_A(A *a) { delete a; }
// CHECK-LABEL: define {{.*}}delete_A
// CHECK: %[[a:.*]] = load
// CHECK: icmp eq %{{.*}} %[[a]], null
// CHECK: br i1
//
// Ensure that we call the destroying delete and not the destructor.
// CHECK-NOT: call
// CHECK-ITANIUM: call void @_ZN1AdlEPS_St19destroying_delete_t(%{{.*}}* noundef %[[a]])
// CHECK-MSABI64: call void @"??3A@@SAXPEAU0@Udestroying_delete_t@std@@@Z"(%{{.*}}* noundef %[[a]], i8
// CHECK-MSABI32: call void @"??3A@@SAXPAU0@Udestroying_delete_t@std@@@Z"(%{{.*}}* noundef %[[a]], %{{.*}}* noundef byval(%{{.*}}) align 4 %{{.*}})
// CHECK-NOT: call
// CHECK: }

struct B {
  virtual ~B();
  void operator delete(B*, std::destroying_delete_t);
};
void delete_B(B *b) { delete b; }
// CHECK-LABEL: define {{.*}}delete_B
// CHECK: %[[b:.*]] = load
// CHECK: icmp eq %{{.*}} %[[b]], null
// CHECK: br i1
//
// Ensure that we call the virtual destructor and not the operator delete.
// CHECK-NOT: call
// CHECK: %[[VTABLE:.*]] = load
// CHECK: %[[DTOR:.*]] = load
// CHECK: call {{void|noundef i8\*|x86_thiscallcc noundef i8\*}} %[[DTOR]](%{{.*}}* {{[^,]*}} %[[b]]
// CHECK-MSABI-SAME: , i32 noundef 1)
// CHECK-NOT: call
// CHECK: }

struct Padding {
  virtual void f();
};

struct C : Padding, A {};
void delete_C(C *c) { delete c; }
// Check that we perform a derived-to-base conversion on the parameter to 'operator delete'.
// CHECK-LABEL: define {{.*}}delete_C
// CHECK: %[[c:.*]] = load
// CHECK: icmp eq %{{.*}} %[[c]], null
// CHECK: br i1
//
// CHECK-64BIT: %[[base:.*]] = getelementptr {{.*}}, i64 8
// CHECK-32BIT: %[[base:.*]] = getelementptr {{.*}}, i32 4
// CHECK: %[[castbase:.*]] = bitcast {{.*}} %[[base]]
//
// CHECK: %[[a:.*]] = phi {{.*}} %[[castbase]]
// CHECK: icmp eq %{{.*}} %[[a]], null
// CHECK: br i1
//
// CHECK-NOT: call
// CHECK-ITANIUM: call void @_ZN1AdlEPS_St19destroying_delete_t(%{{.*}}* noundef %[[a]])
// CHECK-MSABI64: call void @"??3A@@SAXPEAU0@Udestroying_delete_t@std@@@Z"(%{{.*}}* noundef %[[a]], i8
// CHECK-MSABI32: call void @"??3A@@SAXPAU0@Udestroying_delete_t@std@@@Z"(%{{.*}}* noundef %[[a]], %{{.*}}* noundef byval(%{{.*}}) align 4 %{{.*}})
// CHECK-NOT: call
// CHECK: }

struct VDel { virtual ~VDel(); };
struct D : Padding, VDel, B {};
void delete_D(D *d) { delete d; }
// CHECK-LABEL: define {{.*}}delete_D
// CHECK: %[[d:.*]] = load
// CHECK: icmp eq %{{.*}} %[[d]], null
// CHECK: br i1
//
// CHECK-NOT: call
// For MS, we don't add a new vtable slot to the primary vtable for the virtual
// destructor. Instead we cast to the VDel base class.
// CHECK-MSABI: bitcast {{.*}} %[[d]]
// CHECK-MSABI64-NEXT: getelementptr {{.*}}, i64 8
// CHECK-MSABI32-NEXT: getelementptr {{.*}}, i32 4
// CHECK-MSABI-NEXT: %[[d:.*]] = bitcast i8*
//
// CHECK: %[[VTABLE:.*]] = load
// CHECK: %[[DTOR:.*]] = load
//
// CHECK: call {{void|noundef i8\*|x86_thiscallcc noundef i8\*}} %[[DTOR]](%{{.*}}* {{[^,]*}} %[[d]]
// CHECK-MSABI-SAME: , i32 noundef 1)
// CHECK-NOT: call
// CHECK: }

struct J {
  J(); // might throw
  void operator delete(J *, std::destroying_delete_t);
};

// CHECK-ITANIUM-LABEL: define {{.*}}@_Z1j
// CHECK-MSABI-LABEL: define {{.*}}@"?j@@
J *j() {
  // CHECK-ITANIUM: invoke {{.*}}@_ZN1JC1Ev(
  // CHECK-ITANIUM: call {{.*}}@_ZdlPv(
  // CHECK-NOT: }
  // CHECK-MSABI: invoke {{.*}}@"??0J@@Q{{AE|EAA}}@XZ"(
  // CHECK-MSABI: call {{.*}}@"??3@YAXP{{E?}}AX@Z"(
  return new J;
  // CHECK: }
}

struct K {
  K(); // might throw
  void operator delete(void *);
  void operator delete(K *, std::destroying_delete_t);
};

// CHECK-ITANIUM-LABEL: define {{.*}}@_Z1k
// CHECK-MSABI-LABEL: define {{.*}}@"?k@@
K *k() {
  // CHECK-ITANIUM: invoke {{.*}}@_ZN1KC1Ev(
  // CHECK-ITANIUM: call {{.*}}@_ZN1KdlEPv(
  // CHECK-NOT: }
  // CHECK-MSABI: invoke {{.*}}@"??0K@@Q{{AE|EAA}}@XZ"(
  // CHECK-MSABI: call {{.*}}@"??3K@@SAXP{{E?}}AX@Z"(
  return new K;
  // CHECK: }
}

struct E { void *data; };
struct F { void operator delete(F *, std::destroying_delete_t, std::size_t, std::align_val_t); void *data; };
struct alignas(16) G : E, F { void *data; };

void delete_G(G *g) { delete g; }
// CHECK-LABEL: define {{.*}}delete_G
// CHECK-NOT: call
// CHECK-ITANIUM: call void @_ZN1FdlEPS_St19destroying_delete_tmSt11align_val_t(%{{.*}}* noundef %[[a]], i64 noundef 32, i64 noundef 16)
// CHECK-MSABI64: call void @"??3F@@SAXPEAU0@Udestroying_delete_t@std@@_KW4align_val_t@2@@Z"(%{{.*}}* noundef %[[a]], i8 {{[^,]*}}, i64 noundef 32, i64 noundef 16)
// CHECK-MSABI32: call void @"??3F@@SAXPAU0@Udestroying_delete_t@std@@IW4align_val_t@2@@Z"(%{{.*}}* noundef %[[a]], %{{.*}}* noundef byval(%{{.*}}) align 4 %{{.*}}, i32 noundef 16, i32 noundef 16)
// CHECK-NOT: call
// CHECK: }

void call_in_dtor();

struct H : G { virtual ~H(); } h;
H::~H() { call_in_dtor(); }
// CHECK-ITANIUM-LABEL: define{{.*}} void @_ZN1HD0Ev(
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: getelementptr {{.*}}, i64 24
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: call void @_ZN1FdlEPS_St19destroying_delete_tmSt11align_val_t({{.*}}, i64 noundef 48, i64 noundef 16)
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: }

// CHECK-MSABI64-LABEL: define {{.*}} @"??_GH@@UEAAPEAXI@Z"(
// CHECK-MSABI32-LABEL: define {{.*}} @"??_GH@@UAEPAXI@Z"(
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI: load i32
// CHECK-MSABI: icmp eq i32 {{.*}}, 0
// CHECK-MSABI: br i1
//
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI64: getelementptr {{.*}}, i64 24
// CHECK-MSABI32: getelementptr {{.*}}, i32 20
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI64: call void @"??3F@@SAXPEAU0@Udestroying_delete_t@std@@_KW4align_val_t@2@@Z"({{.*}}, i64 noundef 48, i64 noundef 16)
// CHECK-MSABI32: call void @"??3F@@SAXPAU0@Udestroying_delete_t@std@@IW4align_val_t@2@@Z"({{.*}}, i32 noundef 32, i32 noundef 16)
// CHECK-MSABI: br label %[[RETURN:.*]]
//
// CHECK-MSABI64: call void @"??1H@@UEAA@XZ"(
// CHECK-MSABI32: call x86_thiscallcc void @"??1H@@UAE@XZ"(
// CHECK-MSABI: br label %[[RETURN]]
//
// CHECK-MSABI: }

struct I : H { virtual ~I(); alignas(32) char buffer[32]; } i;
I::~I() { call_in_dtor(); }
// CHECK-ITANIUM-LABEL: define{{.*}} void @_ZN1ID0Ev(
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: getelementptr {{.*}}, i64 24
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: call void @_ZN1FdlEPS_St19destroying_delete_tmSt11align_val_t({{.*}}, i64 noundef 96, i64 noundef 32)
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: }

// CHECK-MSABI64-LABEL: define {{.*}} @"??_GI@@UEAAPEAXI@Z"(
// CHECK-MSABI32-LABEL: define {{.*}} @"??_GI@@UAEPAXI@Z"(
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI: load i32
// CHECK-MSABI: icmp eq i32 {{.*}}, 0
// CHECK-MSABI: br i1
//
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI64: getelementptr {{.*}}, i64 24
// CHECK-MSABI32: getelementptr {{.*}}, i32 20
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI64: call void @"??3F@@SAXPEAU0@Udestroying_delete_t@std@@_KW4align_val_t@2@@Z"({{.*}}, i64 noundef 96, i64 noundef 32)
// CHECK-MSABI32: call void @"??3F@@SAXPAU0@Udestroying_delete_t@std@@IW4align_val_t@2@@Z"({{.*}}, i32 noundef 64, i32 noundef 32)
// CHECK-MSABI: br label %[[RETURN:.*]]
//
// CHECK-MSABI64: call void @"??1I@@UEAA@XZ"(
// CHECK-MSABI32: call x86_thiscallcc void @"??1I@@UAE@XZ"(
// CHECK-MSABI: br label %[[RETURN]]
//
// CHECK-MSABI: }
