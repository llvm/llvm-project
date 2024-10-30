// Diamond virtual inheritance.
// This should cover virtual inheritance, construction vtables, and VTTs.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s

// Class A contains a vtable ptr, then int, then padding

// VTable for B. Contains an extra field at the start for the virtual-base offset.
// CHECK: @_ZTV1B.local = internal unnamed_addr constant { [4 x i32], [4 x i32] } { [4 x i32] [i32 8, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B4barBEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -8, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 1, i32 3) to i64)) to i32)] }, align 4

// VTT for B
// CHECK: @_ZTT1B ={{.*}} unnamed_addr constant [2 x ptr] [ptr getelementptr inbounds inrange(-12, 4) ({ [4 x i32], [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 3), ptr getelementptr inbounds inrange(-12, 4) ({ [4 x i32], [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 1, i32 3)], align 8

// VTable for C
// CHECK: @_ZTV1C.local = internal unnamed_addr constant { [4 x i32], [4 x i32] } { [4 x i32] [i32 8, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTV1C.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1C4barCEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTV1C.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -8, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTV1C.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTV1C.local, i32 0, i32 1, i32 3) to i64)) to i32)] }, align 4

// VTT for C
// CHECK: @_ZTT1C ={{.*}} unnamed_addr constant [2 x ptr] [ptr getelementptr inbounds inrange(-12, 4) ({ [4 x i32], [4 x i32] }, ptr @_ZTV1C.local, i32 0, i32 0, i32 3), ptr getelementptr inbounds inrange(-12, 4) ({ [4 x i32], [4 x i32] }, ptr @_ZTV1C.local, i32 0, i32 1, i32 3)], align 8

// VTable for D
// CHECK: @_ZTV1D.local = internal unnamed_addr constant { [5 x i32], [4 x i32], [4 x i32] } { [5 x i32] [i32 16, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1D.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B4barBEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1D3bazEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 8, i32 -8, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1D.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1C4barCEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 1, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -16, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1D.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 2, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 2, i32 3) to i64)) to i32)] }, align 4

// VTT for D
// CHECK: @_ZTT1D ={{.*}} unnamed_addr constant [7 x ptr] [ptr getelementptr inbounds inrange(-12, 8) ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 0, i32 3), ptr getelementptr inbounds inrange(-12, 4) ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D0_1B.local, i32 0, i32 0, i32 3), ptr getelementptr inbounds inrange(-12, 4) ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D0_1B.local, i32 0, i32 1, i32 3), ptr getelementptr inbounds inrange(-12, 4) ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D8_1C.local, i32 0, i32 0, i32 3), ptr getelementptr inbounds inrange(-12, 4) ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D8_1C.local, i32 0, i32 1, i32 3), ptr getelementptr inbounds inrange(-12, 4) ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 2, i32 3), ptr getelementptr inbounds inrange(-12, 4) ({ [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local, i32 0, i32 1, i32 3)], align 8

// Construction vtable for B-in-D
// CHECK: @_ZTC1D0_1B.local = internal unnamed_addr constant { [4 x i32], [4 x i32] } { [4 x i32] [i32 16, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D0_1B.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B4barBEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D0_1B.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -16, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D0_1B.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D0_1B.local, i32 0, i32 1, i32 3) to i64)) to i32)] }, align 4

// Construction vtable for C-in-D
// CHECK: @_ZTC1D8_1C.local = internal unnamed_addr constant { [4 x i32], [4 x i32] } { [4 x i32] [i32 8, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D8_1C.local, i32 0, i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1C4barCEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D8_1C.local, i32 0, i32 0, i32 3) to i64)) to i32)], [4 x i32] [i32 0, i32 -8, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D8_1C.local, i32 0, i32 1, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32], [4 x i32] }, ptr @_ZTC1D8_1C.local, i32 0, i32 1, i32 3) to i64)) to i32)] }, align 4

// CHECK: @_ZTV1B ={{.*}} unnamed_addr alias { [4 x i32], [4 x i32] }, ptr @_ZTV1B.local
// CHECK: @_ZTV1C ={{.*}} unnamed_addr alias { [4 x i32], [4 x i32] }, ptr @_ZTV1C.local
// CHECK: @_ZTC1D0_1B ={{.*}} unnamed_addr alias { [4 x i32], [4 x i32] }, ptr @_ZTC1D0_1B.local
// CHECK: @_ZTC1D8_1C ={{.*}} unnamed_addr alias { [4 x i32], [4 x i32] }, ptr @_ZTC1D8_1C.local
// CHECK: @_ZTV1D ={{.*}} unnamed_addr alias { [5 x i32], [4 x i32], [4 x i32] }, ptr @_ZTV1D.local

// CHECK:      define{{.*}} void @_Z5D_fooP1D(ptr noundef %d) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[vtable:%[a-z0-9]+]] = load ptr, ptr %d, align 8

// This normally would've been -24 (8 bytes for the virtual call/base offset, 8
// bytes for the offset to top, and 8 bytes for the RTTI pointer), but since all
// components in the vtable are halved to 4 bytes, the offset is halved to -12.
// CHECK-NEXT:   [[vbase_offset_ptr:%[a-z0-9.]+]] = getelementptr i8, ptr [[vtable]], i64 -12

// CHECK-NEXT:   [[vbase_offset:%[a-z0-9.]+]] = load i32, ptr [[vbase_offset_ptr]], align 4
// CHECK-NEXT:   [[vbase_offset2:%.+]] = sext i32 [[vbase_offset]] to i64
// CHECK-NEXT:   [[add_ptr:%[a-z0-9.]+]] = getelementptr inbounds i8, ptr %d, i64 [[vbase_offset2]]
// CHECK-NEXT:   [[vtable:%[a-z0-9]+]] = load ptr, ptr [[add_ptr]], align 8
// CHECK-NEXT:   [[ptr:%[0-9]+]] = tail call ptr @llvm.load.relative.i32(ptr [[vtable]], i32 0)
// CHECK-NEXT:   call void [[ptr]](ptr {{[^,]*}} [[add_ptr]])
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
