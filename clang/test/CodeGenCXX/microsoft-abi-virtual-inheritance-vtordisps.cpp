// RUN: %clang_cc1 -no-opaque-pointers %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -o - | FileCheck %s

// For now, just make sure x86_64 doesn't crash.
// RUN: %clang_cc1 -no-opaque-pointers %s -fno-rtti -triple=x86_64-pc-win32 -emit-llvm -o %t

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
// CHECK: %[[COERCE_LOAD:.*]] = load %struct.D*, %struct.D** %{{.*}}
// CHECK: %[[ECX:.*]] = load %struct.D*, %struct.D** %{{.*}}
// CHECK: %[[ECX_i8:.*]] = bitcast %struct.D* %[[ECX]] to i8*
// CHECK: %[[VTORDISP_PTR_i8:.*]] = getelementptr inbounds i8, i8* %[[ECX_i8]], i32 -4
// CHECK: %[[VTORDISP_PTR:.*]] = bitcast i8* %[[VTORDISP_PTR_i8]] to i32*
// CHECK: %[[VTORDISP:.*]] = load i32, i32* %[[VTORDISP_PTR]]
// CHECK: %[[VTORDISP_NEG:.*]] = sub i32 0, %[[VTORDISP]]
// CHECK: %[[ADJUSTED_i8:.*]] = getelementptr i8, i8* %[[ECX_i8]], i32 %[[VTORDISP_NEG]]
// CHECK: call x86_thiscallcc void @"?f@D@@UAEXXZ"(i8* noundef %[[ADJUSTED_i8]])
// CHECK: ret void

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"?f@D@@$4PPPPPPPI@3AEXXZ"
// CHECK: %[[COERCE_LOAD:.*]] = load %struct.D*, %struct.D** %{{.*}}
// CHECK: %[[ECX:.*]] = load %struct.D*, %struct.D** %{{.*}}
// CHECK: %[[ECX_i8:.*]] = bitcast %struct.D* %[[ECX]] to i8*
// CHECK: %[[VTORDISP_PTR_i8:.*]] = getelementptr inbounds i8, i8* %[[ECX_i8]], i32 -8
// CHECK: %[[VTORDISP_PTR:.*]] = bitcast i8* %[[VTORDISP_PTR_i8]] to i32*
// CHECK: %[[VTORDISP:.*]] = load i32, i32* %[[VTORDISP_PTR]]
// CHECK: %[[VTORDISP_NEG:.*]] = sub i32 0, %[[VTORDISP]]
// CHECK: %[[VTORDISP_ADJUSTED_i8:.*]] = getelementptr i8, i8* %[[ECX_i8]], i32 %[[VTORDISP_NEG]]
// CHECK: %[[ADJUSTED_i8:.*]] = getelementptr i8, i8* %[[VTORDISP_ADJUSTED_i8]], i32 -4
// CHECK: call x86_thiscallcc void @"?f@D@@UAEXXZ"(i8* noundef %[[ADJUSTED_i8]])
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

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"?f@E@@$R4BA@M@PPPPPPPM@7AEXXZ"(i8*
// CHECK: %[[COERCE_LOAD:.*]] = load %struct.E*, %struct.E** %{{.*}}
// CHECK: %[[ECX:.*]] = load %struct.E*, %struct.E** %{{.*}}
// CHECK: %[[ECX_i8:.*]] = bitcast %struct.E* %[[ECX]] to i8*
// CHECK: %[[VTORDISP_PTR_i8:.*]] = getelementptr inbounds i8, i8* %[[ECX_i8]], i32 -4
// CHECK: %[[VTORDISP_PTR:.*]] = bitcast i8* %[[VTORDISP_PTR_i8]] to i32*
// CHECK: %[[VTORDISP:.*]] = load i32, i32* %[[VTORDISP_PTR]]
// CHECK: %[[VTORDISP_NEG:.*]] = sub i32 0, %[[VTORDISP]]
// CHECK: %[[VTORDISP_ADJUSTED_i8:.*]] = getelementptr i8, i8* %[[ECX_i8]], i32 %[[VTORDISP_NEG]]
// CHECK: %[[VBPTR_i8:.*]] = getelementptr inbounds i8, i8* %[[VTORDISP_ADJUSTED_i8]], i32 -16
// CHECK: %[[VBPTR:.*]] = bitcast i8* %[[VBPTR_i8]] to i32**
// CHECK: %[[VBTABLE:.*]] = load i32*, i32** %[[VBPTR]]
// CHECK: %[[VBOFFSET_PTR:.*]] = getelementptr inbounds i32, i32* %[[VBTABLE]], i32 3
// CHECK: %[[VBASE_OFFSET:.*]] = load i32, i32* %[[VBOFFSET_PTR]]
// CHECK: %[[VBASE:.*]] = getelementptr inbounds i8, i8* %[[VBPTR_i8]], i32 %[[VBASE_OFFSET]]
// CHECK: %[[ARG_i8:.*]] = getelementptr i8, i8* %[[VBASE]], i32 8
// CHECK: call x86_thiscallcc void @"?f@E@@UAEXXZ"(i8* noundef %[[ARG_i8]])
// CHECK: ret void
