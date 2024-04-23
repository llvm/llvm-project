// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -std=c++11 -emit-llvm -o - | FileCheck %s

// This is the sample from the C++ Itanium ABI, p2.6.2.
namespace Test {
  class A1 { int i; };
  class A2 { int i; virtual void f(); };
  class V1 : public A1, public A2 { int i; };
  class B1 { int i; };
  class B2 { int i; };
  class V2 : public B1, public B2, public virtual V1 { int i; };
  class V3 { virtual void g(); };
  class C1 : public virtual V1 { int i; };
  class C2 : public virtual V3, virtual V2 { int i; };
  class X1 { int i; };
  class C3 : public X1 { int i; };
  class D : public C1, public C2, public C3 { int i;  };

  D d;
}

// CHECK: linkonce_odr unnamed_addr addrspace(1) constant [13 x ptr addrspace(1)] [ptr addrspace(1) getelementptr inbounds inrange(-40, 0) ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN4Test1DE, i32 0, i32 0, i32 5), ptr addrspace(1) getelementptr inbounds inrange(-24, 0) ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN4Test1DE0_NS_2C1E, i32 0, i32 0, i32 3), ptr addrspace(1) getelementptr inbounds inrange(-24, 8) ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN4Test1DE0_NS_2C1E, i32 0, i32 1, i32 3), ptr addrspace(1) getelementptr inbounds inrange(-48, 8) ({ [7 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN4Test1DE16_NS_2C2E, i32 0, i32 0, i32 6), ptr addrspace(1) getelementptr inbounds inrange(-48, 8) ({ [7 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN4Test1DE16_NS_2C2E, i32 0, i32 0, i32 6), ptr addrspace(1) getelementptr inbounds inrange(-24, 0) ({ [7 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN4Test1DE16_NS_2C2E, i32 0, i32 1, i32 3), ptr addrspace(1) getelementptr inbounds inrange(-24, 8) ({ [7 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN4Test1DE16_NS_2C2E, i32 0, i32 2, i32 3), ptr addrspace(1) getelementptr inbounds inrange(-24, 8) ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN4Test1DE, i32 0, i32 2, i32 3), ptr addrspace(1) getelementptr inbounds inrange(-48, 8) ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN4Test1DE, i32 0, i32 1, i32 6), ptr addrspace(1) getelementptr inbounds inrange(-48, 8) ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN4Test1DE, i32 0, i32 1, i32 6), ptr addrspace(1) getelementptr inbounds inrange(-24, 0) ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN4Test1DE, i32 0, i32 3, i32 3), ptr addrspace(1) getelementptr inbounds inrange(-24, 0) ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN4Test1DE64_NS_2V2E, i32 0, i32 0, i32 3), ptr addrspace(1) getelementptr inbounds inrange(-24, 8) ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN4Test1DE64_NS_2V2E, i32 0, i32 1, i32 3)], comdat, align 8
// CHECK: call void @_ZN4Test2V2C2Ev(ptr noundef nonnull align 8 dereferenceable(20) %2, ptr addrspace(1) noundef getelementptr inbounds ([13 x ptr addrspace(1)], ptr addrspace(1) @_ZTTN4Test1DE, i64 0, i64 11))
// CHECK: call void @_ZN4Test2C1C2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this1, ptr addrspace(1) noundef getelementptr inbounds ([13 x ptr addrspace(1)], ptr addrspace(1) @_ZTTN4Test1DE, i64 0, i64 1))
// CHECK: call void @_ZN4Test2C2C2Ev(ptr noundef nonnull align 8 dereferenceable(12) %3, ptr addrspace(1) noundef getelementptr inbounds ([13 x ptr addrspace(1)], ptr addrspace(1) @_ZTTN4Test1DE, i64 0, i64 3))
// CHECK: define linkonce_odr void @_ZN4Test2V2C2Ev(ptr noundef nonnull align 8 dereferenceable(20) %this, ptr addrspace(1) noundef %vtt)
// CHECK: define linkonce_odr void @_ZN4Test2C1C2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this, ptr addrspace(1) noundef %vtt)
// CHECK: define linkonce_odr void @_ZN4Test2C2C2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this, ptr addrspace(1) noundef %vtt)
