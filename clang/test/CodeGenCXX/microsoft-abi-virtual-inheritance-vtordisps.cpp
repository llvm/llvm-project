// RUN: %clang_cc1 %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -o - | FileCheck %s

// For now, just make sure x86_64 doesn't crash.
// RUN: %clang_cc1 %s -fno-rtti -triple=x86_64-pc-win32 -emit-llvm -o %t

struct A {
  virtual void f();
};

struct B {
  virtual void f();
};

struct C : A, B {};

struct D : virtual C {
  D();
  ~D();
  virtual void f();
  void g();
  int xxx;
};

D::D() {}  // Forces vftable emission.

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"?f@D@@$4PPPPPPPM@A@AEXXZ"
// Note that the vtordisp is applied before really adjusting to D*.
// CHECK: %[[ECX:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[VTORDISP_PTR_i8:.*]] = getelementptr inbounds i8, ptr %[[ECX]], i32 -4
// CHECK: %[[VTORDISP:.*]] = load i32, ptr %[[VTORDISP_PTR_i8]]
// CHECK: %[[VTORDISP_NEG:.*]] = sub i32 0, %[[VTORDISP]]
// CHECK: %[[ADJUSTED_i8:.*]] = getelementptr i8, ptr %[[ECX]], i32 %[[VTORDISP_NEG]]
// CHECK: call x86_thiscallcc void @"?f@D@@UAEXXZ"(ptr noundef %[[ADJUSTED_i8]])
// CHECK: ret void

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"?f@D@@$4PPPPPPPI@3AEXXZ"
// CHECK: %[[ECX:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[VTORDISP_PTR_i8:.*]] = getelementptr inbounds i8, ptr %[[ECX]], i32 -8
// CHECK: %[[VTORDISP:.*]] = load i32, ptr %[[VTORDISP_PTR_i8]]
// CHECK: %[[VTORDISP_NEG:.*]] = sub i32 0, %[[VTORDISP]]
// CHECK: %[[VTORDISP_ADJUSTED_i8:.*]] = getelementptr i8, ptr %[[ECX]], i32 %[[VTORDISP_NEG]]
// CHECK: %[[ADJUSTED_i8:.*]] = getelementptr i8, ptr %[[VTORDISP_ADJUSTED_i8]], i32 -4
// CHECK: call x86_thiscallcc void @"?f@D@@UAEXXZ"(ptr noundef %[[ADJUSTED_i8]])
// CHECK: ret void

struct E : virtual A {
  virtual void f();
  ~E();
};

struct F {
  virtual void z();
};

struct G : virtual F, virtual E {
  int ggg;
  G();
  ~G();
};

G::G() {}  // Forces vftable emission.

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"?f@E@@$R4BA@M@PPPPPPPM@7AEXXZ"(ptr
// CHECK: %[[ECX:.*]] = load ptr, ptr %{{.*}}
// CHECK: %[[VTORDISP_PTR_i8:.*]] = getelementptr inbounds i8, ptr %[[ECX]], i32 -4
// CHECK: %[[VTORDISP:.*]] = load i32, ptr %[[VTORDISP_PTR_i8]]
// CHECK: %[[VTORDISP_NEG:.*]] = sub i32 0, %[[VTORDISP]]
// CHECK: %[[VTORDISP_ADJUSTED_i8:.*]] = getelementptr i8, ptr %[[ECX]], i32 %[[VTORDISP_NEG]]
// CHECK: %[[VBPTR_i8:.*]] = getelementptr inbounds i8, ptr %[[VTORDISP_ADJUSTED_i8]], i32 -16
// CHECK: %[[VBTABLE:.*]] = load ptr, ptr %[[VBPTR_i8]]
// CHECK: %[[VBOFFSET_PTR:.*]] = getelementptr inbounds i32, ptr %[[VBTABLE]], i32 3
// CHECK: %[[VBASE_OFFSET:.*]] = load i32, ptr %[[VBOFFSET_PTR]]
// CHECK: %[[VBASE:.*]] = getelementptr inbounds i8, ptr %[[VBPTR_i8]], i32 %[[VBASE_OFFSET]]
// CHECK: %[[ARG_i8:.*]] = getelementptr i8, ptr %[[VBASE]], i32 8
// CHECK: call x86_thiscallcc void @"?f@E@@UAEXXZ"(ptr noundef %[[ARG_i8]])
// CHECK: ret void
