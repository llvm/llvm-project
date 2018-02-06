// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -triple x86_64-linux-gnu -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ITANIUM
// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -triple x86_64-windows -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-MSABI

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
// CHECK-ITANIUM: call void @_ZN1AdlEPS_St19destroying_delete_t(%{{.*}}* %[[a]])
// CHECK-MSABI: call void @"\01??3A@@SAXPEAU0@Udestroying_delete_t@std@@@Z"(%{{.*}}* %[[a]], i8
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
// CHECK: call {{void|i8\*}} %[[DTOR]](%{{.*}}* %[[b]]
// CHECK-MSABI-SAME: , i32 1)
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
// CHECK: %[[base:.*]] = getelementptr {{.*}}, i64 8
// CHECK: %[[castbase:.*]] = bitcast {{.*}} %[[base]]
//
// CHECK: %[[a:.*]] = phi {{.*}} %[[castbase]]
// CHECK: icmp eq %{{.*}} %[[a]], null
// CHECK: br i1
//
// CHECK-NOT: call
// CHECK-ITANIUM: call void @_ZN1AdlEPS_St19destroying_delete_t(%{{.*}}* %[[a]])
// CHECK-MSABI: call void @"\01??3A@@SAXPEAU0@Udestroying_delete_t@std@@@Z"(%{{.*}}* %[[a]], i8
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
// CHECK-MSABI-NEXT: getelementptr {{.*}}, i64 8
// CHECK-MSABI-NEXT: %[[d:.*]] = bitcast i8*
//
// CHECK: %[[VTABLE:.*]] = load
// CHECK: %[[DTOR:.*]] = load
//
// CHECK: call {{void|i8\*}} %[[DTOR]](%{{.*}}* %[[d]]
// CHECK-MSABI-SAME: , i32 1)
// CHECK-NOT: call
// CHECK: }

struct E { void *data; };
struct F { void operator delete(F *, std::destroying_delete_t, std::size_t, std::align_val_t); void *data; };
struct alignas(16) G : E, F { void *data; };

void delete_G(G *g) { delete g; }
// CHECK-LABEL: define {{.*}}delete_G
// CHECK-NOT: call
// CHECK-ITANIUM: call void @_ZN1FdlEPS_St19destroying_delete_tmSt11align_val_t(%{{.*}}* %[[a]], i64 32, i64 16)
// CHECK-MSABI: call void @"\01??3F@@SAXPEAU0@Udestroying_delete_t@std@@_KW4align_val_t@2@@Z"(%{{.*}}* %[[a]], i8 {{[^,]*}}, i64 32, i64 16)
// CHECK-NOT: call
// CHECK: }

void call_in_dtor();

struct H : G { virtual ~H(); } h;
H::~H() { call_in_dtor(); }
// CHECK-ITANIUM-LABEL: define void @_ZN1HD0Ev(
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: getelementptr {{.*}}, i64 24
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: call void @_ZN1FdlEPS_St19destroying_delete_tmSt11align_val_t({{.*}}, i64 48, i64 16)
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: }

// CHECK-MSABI: define {{.*}} @"\01??_GH@@UEAAPEAXI@Z"(
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI: load i32
// CHECK-MSABI: icmp eq i32 {{.*}}, 0
// CHECK-MSABI: br i1
//
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI: getelementptr {{.*}}, i64 24
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI: call void @"\01??3F@@SAXPEAU0@Udestroying_delete_t@std@@_KW4align_val_t@2@@Z"({{.*}}, i64 48, i64 16)
// CHECK-MSABI: br label %[[RETURN:.*]]
//
// CHECK-MSABI: call void @"\01??_DH@@QEAAXXZ"(
// CHECK-MSABI: br label %[[RETURN]]
//
// CHECK-MSABI: }

struct I : H { virtual ~I(); alignas(32) char buffer[32]; } i;
I::~I() { call_in_dtor(); }
// CHECK-ITANIUM-LABEL: define void @_ZN1ID0Ev(
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: getelementptr {{.*}}, i64 24
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: call void @_ZN1FdlEPS_St19destroying_delete_tmSt11align_val_t({{.*}}, i64 96, i64 32)
// CHECK-ITANIUM-NOT: call
// CHECK-ITANIUM: }

// CHECK-MSABI: define {{.*}} @"\01??_GI@@UEAAPEAXI@Z"(
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI: load i32
// CHECK-MSABI: icmp eq i32 {{.*}}, 0
// CHECK-MSABI: br i1
//
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI: getelementptr {{.*}}, i64 24
// CHECK-MSABI-NOT: call{{ }}
// CHECK-MSABI: call void @"\01??3F@@SAXPEAU0@Udestroying_delete_t@std@@_KW4align_val_t@2@@Z"({{.*}}, i64 96, i64 32)
// CHECK-MSABI: br label %[[RETURN:.*]]
//
// CHECK-MSABI: call void @"\01??_DI@@QEAAXXZ"(
// CHECK-MSABI: br label %[[RETURN]]
//
// CHECK-MSABI: }
