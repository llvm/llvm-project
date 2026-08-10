// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -std=c++11 | FileCheck %s

struct A { 
  A();
  A(const A&);
  A(A&);
  ~A();
};

struct B {
  B();
  B(B&);
};

struct C {
  C() {}
  C(C& other, A a = A());
  int i, j;
};

struct POD {
  int array[3][4];
};

struct D : A, B, virtual C { 
  D();
  int scalar;
  int scalar_array[2][3];
  B class_member;
  C class_member_array[2][3];
  POD pod_array[2][3];

  union {
    int x;
    float f[3];
  };
};

void f(D d) {
  D d2(d);
}

// CHECK-LABEL: define linkonce_odr void @_ZN1DC1ERS_(ptr {{[^,]*}} %this, ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0) unnamed_addr
// CHECK: call void @_ZN1AC1Ev
// CHECK: call void @_ZN1CC2ERS_1A
// CHECK: call void @_ZN1AD1Ev
// CHECK: call void @_ZN1AC2ERS_
// CHECK: call void @_ZN1BC2ERS_
// CHECK: {{call void @llvm.memcpy.p0.p0.i64.*i64 28}}
// CHECK: call void @_ZN1BC1ERS_
// CHECK: br label
// CHECK: call void @_ZN1AC1Ev
// CHECK: call void @_ZN1CC1ERS_1A
// CHECK: call void @_ZN1AD1Ev
// CHECK: {{icmp eq.*, 3}}
// CHECK: br i1
// CHECK: {{icmp eq.*, 2}}
// CHECK: br i1
// CHECK: {{call void @llvm.memcpy.p0.p0.i64.*i64 300}}
// CHECK: ret void


template<class T> struct X0 { void f0(T * ) { } };
template <class > struct X1 { X1( X1& , int = 0 ) { } };
struct X2 { X1<int> result; };
void test_X2()
{
  typedef X2 impl;
  typedef X0<impl> pimpl;
  impl* i;
  pimpl pdata;
  pdata.f0( new impl(*i));
}

namespace test3 {
  struct A { A(const A&); A&operator=(const A&); };
  struct B { A a; unsigned : 0; };
  void test(const B &x) {
    B y = x;
    y = x;
  }
}

namespace test4 {
  // When determining whether to implement an array copy as a memcpy, look at
  // whether the *selected* constructor is trivial.
  struct S {
    int arr[5][5];
    S(S &);
    S(const S &) = default;
  };
  // CHECK: @_ZN5test42f1
  void f1(S a) {
    // CHECK-NOT: memcpy
    // CHECK: call void @_ZN5test41SC1ERS0_
    // CHECK-NOT: memcpy
    S b(a);
    // CHECK: }
  }
  // CHECK: @_ZN5test42f2
  void f2(const S a) {
    // CHECK-NOT: call void @_ZN5test41SC1ERS0_
    // CHECK: memcpy
    // CHECK-NOT: call void @_ZN5test41SC1ERS0_
    S b(a);
    // CHECK: }
  }
}
