// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -std=c++11 -emit-llvm -o - | FileCheck %s

// Test1::B should just have a single entry in its VTT, which points to the vtable.
namespace Test1 {
struct A { };

struct B : virtual A {
  virtual void f();
};

void B::f() { }
}

// Check that we don't add a secondary virtual pointer for Test2::A, since Test2::A doesn't have any virtual member functions or bases.
namespace Test2 {
  struct A { };

  struct B : A { virtual void f(); };
  struct C : virtual B { };

  C c;
}

// This is the sample from the C++ Itanium ABI, p2.6.2.
namespace Test3 {
  class A1 { int i; };
  class A2 { int i; virtual void f(); };
  class V1 : public A1, public A2 { int i; };
  class B1 { int i; };
  class B2 { int i; };
  class V2 : public B1, public B2, public virtual V1 { int i; };
  class V3 {virtual void g(); };
  class C1 : public virtual V1 { int i; };
  class C2 : public virtual V3, virtual V2 { int i; };
  class X1 { int i; };
  class C3 : public X1 { int i; };
  class D : public C1, public C2, public C3 { int i;  };

  D d;
}

// This is the sample from the C++ Itanium ABI, p2.6.2, with the change suggested
// (making A2 a virtual base of V1)
namespace Test4 {
  class A1 { int i; };
  class A2 { int i; virtual void f(); };
  class V1 : public A1, public virtual A2 { int i; };
  class B1 { int i; };
  class B2 { int i; };
  class V2 : public B1, public B2, public virtual V1 { int i; };
  class V3 {virtual void g(); };
  class C1 : public virtual V1 { int i; };
  class C2 : public virtual V3, virtual V2 { int i; };
  class X1 { int i; };
  class C3 : public X1 { int i; };
  class D : public C1, public C2, public C3 { int i;  };

  D d;
}

namespace Test5 {
  struct A {
    virtual void f() = 0;
    virtual void anchor();
  };

  void A::anchor() {
  }
}

namespace Test6 {
  struct A {
    virtual void f() = delete;
    virtual void anchor();
  };

  void A::anchor() {
  }
}

// CHECK: @_ZTTN5Test11BE ={{.*}} unnamed_addr addrspace(1) constant [1 x ptr addrspace(1)] [ptr addrspace(1) getelementptr inbounds ({ [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test11BE, i32 0, inrange i32 0, i32 3)]
// CHECK: @_ZTVN5Test51AE ={{.*}} unnamed_addr addrspace(1) constant { [4 x ptr addrspace(1)] } { [4 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) @_ZTIN5Test51AE, ptr addrspace(1) addrspacecast (ptr @__cxa_pure_virtual to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @_ZN5Test51A6anchorEv to ptr addrspace(1))] }
// CHECK: @_ZTVN5Test61AE ={{.*}} unnamed_addr addrspace(1) constant { [4 x ptr addrspace(1)] } { [4 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) @_ZTIN5Test61AE, ptr addrspace(1) addrspacecast (ptr @__cxa_deleted_virtual to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @_ZN5Test61A6anchorEv to ptr addrspace(1))] }
// CHECK: @_ZTTN5Test21CE = linkonce_odr unnamed_addr addrspace(1) constant [2 x ptr addrspace(1)] [ptr addrspace(1) getelementptr inbounds ({ [5 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test21CE, i32 0, inrange i32 0, i32 4), ptr addrspace(1) getelementptr inbounds ({ [5 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test21CE, i32 0, inrange i32 0, i32 4)]
// CHECK: @_ZTTN5Test31DE = linkonce_odr unnamed_addr addrspace(1) constant [13 x ptr addrspace(1)] [ptr addrspace(1) getelementptr inbounds ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test31DE, i32 0, inrange i32 0, i32 5), ptr addrspace(1) getelementptr inbounds ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test31DE0_NS_2C1E, i32 0, inrange i32 0, i32 3), ptr addrspace(1) getelementptr inbounds ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test31DE0_NS_2C1E, i32 0, inrange i32 1, i32 3), ptr addrspace(1) getelementptr inbounds ({ [7 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test31DE16_NS_2C2E, i32 0, inrange i32 0, i32 6), ptr addrspace(1) getelementptr inbounds ({ [7 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test31DE16_NS_2C2E, i32 0, inrange i32 0, i32 6), ptr addrspace(1) getelementptr inbounds ({ [7 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test31DE16_NS_2C2E, i32 0, inrange i32 1, i32 3), ptr addrspace(1) getelementptr inbounds ({ [7 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test31DE16_NS_2C2E, i32 0, inrange i32 2, i32 3), ptr addrspace(1) getelementptr inbounds ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test31DE, i32 0, inrange i32 2, i32 3), ptr addrspace(1) getelementptr inbounds ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test31DE, i32 0, inrange i32 1, i32 6), ptr addrspace(1) getelementptr inbounds ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test31DE, i32 0, inrange i32 1, i32 6), ptr addrspace(1) getelementptr inbounds ({ [5 x ptr addrspace(1)], [7 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test31DE, i32 0, inrange i32 3, i32 3), ptr addrspace(1) getelementptr inbounds ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test31DE64_NS_2V2E, i32 0, inrange i32 0, i32 3), ptr addrspace(1) getelementptr inbounds ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test31DE64_NS_2V2E, i32 0, inrange i32 1, i32 3)]
// CHECK: @_ZTVN5Test41DE = linkonce_odr unnamed_addr addrspace(1) constant { [6 x ptr addrspace(1)], [8 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)], [4 x ptr addrspace(1)] } { [6 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 72 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 16 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 56 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 40 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) @_ZTIN5Test41DE], [8 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 40 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 24 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 56 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) inttoptr (i64 -16 to ptr addrspace(1)), ptr addrspace(1) @_ZTIN5Test41DE, ptr addrspace(1) addrspacecast (ptr @_ZN5Test42V31gEv to ptr addrspace(1))], [3 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 16 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 -40 to ptr addrspace(1)), ptr addrspace(1) @_ZTIN5Test41DE], [4 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) inttoptr (i64 -56 to ptr addrspace(1)), ptr addrspace(1) @_ZTIN5Test41DE, ptr addrspace(1) addrspacecast (ptr @_ZN5Test42A21fEv to ptr addrspace(1))], [4 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 -16 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 -32 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 -72 to ptr addrspace(1)), ptr addrspace(1) @_ZTIN5Test41DE] }
// CHECK: @_ZTTN5Test41DE = linkonce_odr unnamed_addr addrspace(1) constant [19 x ptr addrspace(1)] [ptr addrspace(1) getelementptr inbounds ({ [6 x ptr addrspace(1)], [8 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test41DE, i32 0, inrange i32 0, i32 6), ptr addrspace(1) getelementptr inbounds ({ [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE0_NS_2C1E, i32 0, inrange i32 0, i32 4), ptr addrspace(1) getelementptr inbounds ({ [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE0_NS_2C1E, i32 0, inrange i32 1, i32 3), ptr addrspace(1) getelementptr inbounds ({ [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE0_NS_2C1E, i32 0, inrange i32 2, i32 3), ptr addrspace(1) getelementptr inbounds ({ [8 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE16_NS_2C2E, i32 0, inrange i32 0, i32 7), ptr addrspace(1) getelementptr inbounds ({ [8 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE16_NS_2C2E, i32 0, inrange i32 0, i32 7), ptr addrspace(1) getelementptr inbounds ({ [8 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE16_NS_2C2E, i32 0, inrange i32 1, i32 4), ptr addrspace(1) getelementptr inbounds ({ [8 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE16_NS_2C2E, i32 0, inrange i32 2, i32 3), ptr addrspace(1) getelementptr inbounds ({ [8 x ptr addrspace(1)], [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE16_NS_2C2E, i32 0, inrange i32 3, i32 3), ptr addrspace(1) getelementptr inbounds ({ [6 x ptr addrspace(1)], [8 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test41DE, i32 0, inrange i32 2, i32 3), ptr addrspace(1) getelementptr inbounds ({ [6 x ptr addrspace(1)], [8 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test41DE, i32 0, inrange i32 3, i32 3), ptr addrspace(1) getelementptr inbounds ({ [6 x ptr addrspace(1)], [8 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test41DE, i32 0, inrange i32 1, i32 7), ptr addrspace(1) getelementptr inbounds ({ [6 x ptr addrspace(1)], [8 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test41DE, i32 0, inrange i32 1, i32 7), ptr addrspace(1) getelementptr inbounds ({ [6 x ptr addrspace(1)], [8 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTVN5Test41DE, i32 0, inrange i32 4, i32 4), ptr addrspace(1) getelementptr inbounds ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE40_NS_2V1E, i32 0, inrange i32 0, i32 3), ptr addrspace(1) getelementptr inbounds ({ [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE40_NS_2V1E, i32 0, inrange i32 1, i32 3), ptr addrspace(1) getelementptr inbounds ({ [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE72_NS_2V2E, i32 0, inrange i32 0, i32 4), ptr addrspace(1) getelementptr inbounds ({ [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE72_NS_2V2E, i32 0, inrange i32 1, i32 3), ptr addrspace(1) getelementptr inbounds ({ [4 x ptr addrspace(1)], [3 x ptr addrspace(1)], [4 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTCN5Test41DE72_NS_2V2E, i32 0, inrange i32 2, i32 3)]
// CHECK: declare void @__cxa_pure_virtual() unnamed_addr
// CHECK: declare void @__cxa_deleted_virtual() unnamed_addr
