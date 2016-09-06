// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - | FileCheck %s

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

// CHECK: @_ZTTN5Test11BE = unnamed_addr constant [1 x i8*] [i8* bitcast (i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTVN5Test11BE, i32 0, i32 3) to i8*)]
// CHECK: @_ZTVN5Test51AE = unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTIN5Test51AE to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void (%"struct.Test5::A"*)* @_ZN5Test51A6anchorEv to i8*)]
// CHECK: @_ZTVN5Test61AE = unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTIN5Test61AE to i8*), i8* bitcast (void ()* @__cxa_deleted_virtual to i8*), i8* bitcast (void (%"struct.Test6::A"*)* @_ZN5Test61A6anchorEv to i8*)]
// CHECK: @_ZTTN5Test21CE = linkonce_odr unnamed_addr constant [2 x i8*] [i8* bitcast (i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @_ZTVN5Test21CE, i32 0, i32 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @_ZTVN5Test21CE, i32 0, i32 4) to i8*)] 
// CHECK: @_ZTTN5Test31DE = linkonce_odr unnamed_addr constant [13 x i8*] [i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTVN5Test31DE, i32 0, i32 5) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*], [7 x i8*]* @_ZTCN5Test31DE0_NS_2C1E, i32 0, i32 3) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*], [7 x i8*]* @_ZTCN5Test31DE0_NS_2C1E, i32 0, i32 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*], [14 x i8*]* @_ZTCN5Test31DE16_NS_2C2E, i32 0, i32 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*], [14 x i8*]* @_ZTCN5Test31DE16_NS_2C2E, i32 0, i32 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*], [14 x i8*]* @_ZTCN5Test31DE16_NS_2C2E, i32 0, i32 10) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*], [14 x i8*]* @_ZTCN5Test31DE16_NS_2C2E, i32 0, i32 13) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTVN5Test31DE, i32 0, i32 15) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTVN5Test31DE, i32 0, i32 11) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTVN5Test31DE, i32 0, i32 11) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTVN5Test31DE, i64 1, i32 0) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*], [7 x i8*]* @_ZTCN5Test31DE64_NS_2V2E, i32 0, i32 3) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*], [7 x i8*]* @_ZTCN5Test31DE64_NS_2V2E, i32 0, i32 6) to i8*)] 
// CHECK: @_ZTTN5Test41DE = linkonce_odr unnamed_addr constant [19 x i8*] [i8* bitcast (i8** getelementptr inbounds ([25 x i8*], [25 x i8*]* @_ZTVN5Test41DE, i32 0, i32 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*], [11 x i8*]* @_ZTCN5Test41DE0_NS_2C1E, i32 0, i32 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*], [11 x i8*]* @_ZTCN5Test41DE0_NS_2C1E, i32 0, i32 7) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*], [11 x i8*]* @_ZTCN5Test41DE0_NS_2C1E, i32 0, i32 10) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i32 0, i32 7) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i32 0, i32 7) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i32 0, i32 12) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i32 0, i32 15) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*], [19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i32 0, i32 18) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*], [25 x i8*]* @_ZTVN5Test41DE, i32 0, i32 17) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*], [25 x i8*]* @_ZTVN5Test41DE, i32 0, i32 20) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*], [25 x i8*]* @_ZTVN5Test41DE, i32 0, i32 13) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*], [25 x i8*]* @_ZTVN5Test41DE, i32 0, i32 13) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*], [25 x i8*]* @_ZTVN5Test41DE, i64 1, i32 0) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*], [7 x i8*]* @_ZTCN5Test41DE40_NS_2V1E, i32 0, i32 3) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*], [7 x i8*]* @_ZTCN5Test41DE40_NS_2V1E, i32 0, i32 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*], [11 x i8*]* @_ZTCN5Test41DE72_NS_2V2E, i32 0, i32 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*], [11 x i8*]* @_ZTCN5Test41DE72_NS_2V2E, i32 0, i32 7) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*], [11 x i8*]* @_ZTCN5Test41DE72_NS_2V2E, i32 0, i32 10) to i8*)] 
// CHECK: declare void @__cxa_pure_virtual() unnamed_addr
// CHECK: declare void @__cxa_deleted_virtual() unnamed_addr
