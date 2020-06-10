// Diamond virtual inheritance.
// This should cover virtual inheritance, construction vtables, and VTTs.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

// CHECK: %class.B = type { i32 (...)**, %class.A.base }
// CHECK: %class.A.base = type <{ i32 (...)**, i32 }>
// CHECK: %class.C = type { i32 (...)**, %class.A.base }
// CHECK: %class.D = type { %class.B.base, %class.C.base, %class.A.base }
// CHECK: %class.B.base = type { i32 (...)** }
// CHECK: %class.C.base = type { i32 (...)** }

// Class A contains a vtable ptr, then int, then padding
// CHECK: %class.A = type <{ i32 (...)**, i32, [4 x i8] }>

// VTable for B. Contains an extra field at the start for the virtual-base offset.
// CHECK: @_ZTV1B.local = private unnamed_addr constant { [4 x i32], [4 x i32] } { [4 x i32] [i32 8, i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64 }** @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1B.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.B*)* @_ZN1B4barBEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1B.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -8, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64 }** @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1B.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1B.local, i32 0, i32 1, i32 3) to i64)) to i32)] }, align 4

// VTT for B
// CHECK: @_ZTT1B = unnamed_addr constant [2 x i8*] [i8* bitcast (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1B.local, i32 0, inrange i32 0, i32 3) to i8*), i8* bitcast (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1B.local, i32 0, inrange i32 1, i32 3) to i8*)], align 8

// VTable for C
// CHECK: @_ZTV1C.local = private unnamed_addr constant { [4 x i32], [4 x i32] } { [4 x i32] [i32 8, i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64 }** @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1C.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.C*)* @_ZN1C4barCEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1C.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -8, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64 }** @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1C.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1C.local, i32 0, i32 1, i32 3) to i64)) to i32)] }, align 4

// VTT for C
// CHECK: @_ZTT1C = unnamed_addr constant [2 x i8*] [i8* bitcast (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1C.local, i32 0, inrange i32 0, i32 3) to i8*), i8* bitcast (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1C.local, i32 0, inrange i32 1, i32 3) to i8*)], align 8

// VTable for D
// CHECK: @_ZTV1D.local = private unnamed_addr constant { [5 x i32], [4 x i32], [4 x i32] } { [5 x i32] [i32 16, i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }** @_ZTI1D.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.B*)* @_ZN1B4barBEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.D*)* @_ZN1D3bazEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 8, i32 -8, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }** @_ZTI1D.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.C*)* @_ZN1C4barCEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, i32 1, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -16, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }** @_ZTI1D.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, i32 2, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, i32 2, i32 3) to i64)) to i32)] }, align 4

// VTT for D
// CHECK: @_ZTT1D = unnamed_addr constant [7 x i8*] [i8* bitcast (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, inrange i32 0, i32 3) to i8*), i8* bitcast (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D0_1B.local, i32 0, inrange i32 0, i32 3) to i8*), i8* bitcast (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D0_1B.local, i32 0, inrange i32 1, i32 3) to i8*), i8* bitcast (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D8_1C.local, i32 0, inrange i32 0, i32 3) to i8*), i8* bitcast (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D8_1C.local, i32 0, inrange i32 1, i32 3) to i8*), i8* bitcast (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, inrange i32 2, i32 3) to i8*), i8* bitcast (i32* getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local, i32 0, inrange i32 1, i32 3) to i8*)], align 8

// Construction vtable for B-in-D
// CHECK: @_ZTC1D0_1B.local = private unnamed_addr constant { [4 x i32], [4 x i32] } { [4 x i32] [i32 16, i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64 }** @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D0_1B.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.B*)* @_ZN1B4barBEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D0_1B.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -16, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64 }** @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D0_1B.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D0_1B.local, i32 0, i32 1, i32 3) to i64)) to i32)] }, align 4

// Construction vtable for C-in-D
// CHECK: @_ZTC1D8_1C.local = private unnamed_addr constant { [4 x i32], [4 x i32] } { [4 x i32] [i32 8, i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64 }** @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D8_1C.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.C*)* @_ZN1C4barCEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D8_1C.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -8, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64 }** @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D8_1C.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D8_1C.local, i32 0, i32 1, i32 3) to i64)) to i32)] }, align 4

// CHECK: @_ZTV1B = unnamed_addr alias { [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1B.local
// CHECK: @_ZTV1C = unnamed_addr alias { [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTV1C.local
// CHECK: @_ZTC1D0_1B = unnamed_addr alias { [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D0_1B.local
// CHECK: @_ZTC1D8_1C = unnamed_addr alias { [4 x i32], [4 x i32] }, { [4 x i32], [4 x i32] }* @_ZTC1D8_1C.local
// CHECK: @_ZTV1D = unnamed_addr alias { [5 x i32], [4 x i32], [4 x i32] }, { [5 x i32], [4 x i32], [4 x i32] }* @_ZTV1D.local

// CHECK:      define void @_Z5D_fooP1D(%class.D* %d) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[d:%[0-9]+]] = bitcast %class.D* %d to i8**
// CHECK-NEXT:   [[vtable:%[a-z0-9]+]] = load i8*, i8** [[d]], align 8

// This normally would've been -24 (8 bytes for the virtual call/base offset, 8
// bytes for the offset to top, and 8 bytes for the RTTI pointer), but since all
// components in the vtable are halved to 4 bytes, the offset is halved to -12.
// CHECK-NEXT:   [[vbase_offset_ptr:%[a-z0-9.]+]] = getelementptr i8, i8* [[vtable]], i64 -12

// CHECK-NEXT:   [[vbase_offset_ptr2:%[a-z0-9.]+]] = bitcast i8* [[vbase_offset_ptr]] to i32*
// CHECK-NEXT:   [[vbase_offset:%[a-z0-9.]+]] = load i32, i32* [[vbase_offset_ptr2]], align 4
// CHECK-NEXT:   [[d:%[0-9]+]] = bitcast %class.D* %d to i8*
// CHECK-NEXT:   [[vbase_offset2:%.+]] = sext i32 [[vbase_offset]] to i64
// CHECK-NEXT:   [[add_ptr:%[a-z0-9.]+]] = getelementptr inbounds i8, i8* [[d]], i64 [[vbase_offset2]]
// CHECK-NEXT:   [[a:%[0-9]+]] = bitcast i8* [[add_ptr]] to %class.A*
// CHECK-NEXT:   [[a_i8_ptr:%[0-9]+]] = bitcast i8* [[add_ptr]] to i8**
// CHECK-NEXT:   [[vtable:%[a-z0-9]+]] = load i8*, i8** [[a_i8_ptr]], align 8
// CHECK-NEXT:   [[ptr:%[0-9]+]] = call i8* @llvm.load.relative.i32(i8* [[vtable]], i32 0)
// CHECK-NEXT:   [[method:%[0-9]+]] = bitcast i8* [[ptr]] to void (%class.A*)*
// CHECK-NEXT:   call void [[method]](%class.A* [[a]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

class A {
public:
  virtual void foo();
  int a;
};

class B : public virtual A {
public:
  virtual void barB();
};

class C : public virtual A {
  virtual void barC();
};

class D : public B, C {
public:
  virtual void baz();
};

void B::barB() {}
void C::barC() {}
void D::baz() {}

void D_foo(D *d) {
  d->foo();
}
