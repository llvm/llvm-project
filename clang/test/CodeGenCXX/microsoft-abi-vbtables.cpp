// RUN: %clang_cc1 %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -o - | FileCheck %s

// See microsoft-abi-structors.cpp for constructor codegen tests.

namespace Test1 {
// Classic diamond, fully virtual.
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual A { int c; };
struct D : virtual B, virtual C { int d; };
D d; // Force vbtable emission.

// Layout should be:
// D: vbptr D
//    int d
// A: int a
// B: vbptr B
//    int b
// C: vbptr C
//    int c

// CHECK-DAG: @"??_8D@Test1@@7B01@@" = linkonce_odr unnamed_addr constant [4 x i32] [i32 0, i32 8, i32 12, i32 20]
// CHECK-DAG: @"??_8D@Test1@@7BB@1@@" = {{.*}} [2 x i32] [i32 0, i32 -4]
// CHECK-DAG: @"??_8D@Test1@@7BC@1@@" = {{.*}} [2 x i32] [i32 0, i32 -12]
// CHECK-DAG: @"??_8C@Test1@@7B@" = {{.*}} [2 x i32] [i32 0, i32 8]
// CHECK-DAG: @"??_8B@Test1@@7B@" = {{.*}} [2 x i32] [i32 0, i32 8]
}

namespace Test2 {
// Classic diamond, only A is virtual.
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual A { int c; };
struct D : B, C { int d; };
D d; // Force vbtable emission.

// Layout should be:
// B: vbptr B
//    int b
// C: vbptr C
//    int c
// D: int d
// A: int a

// CHECK-DAG: @"??_8D@Test2@@7BB@1@@" = {{.*}} [2 x i32] [i32 0, i32 20]
// CHECK-DAG: @"??_8D@Test2@@7BC@1@@" = {{.*}} [2 x i32] [i32 0, i32 12]
// CHECK-DAG: @"??_8C@Test2@@7B@" = {{.*}} [2 x i32] [i32 0, i32 8]
// CHECK-DAG: @"??_8B@Test2@@7B@" = {{.*}} [2 x i32] [i32 0, i32 8]
}

namespace Test3 {
struct A { int a; };
struct B { int b; };
struct C : virtual A, virtual B { int c; };
C c;

// CHECK-DAG: @"??_8C@Test3@@7B@" = {{.*}} [3 x i32] [i32 0, i32 8, i32 12]
}

namespace Test4 {
// Test reusing a vbptr from a non-virtual base.
struct A { int a; };
struct B : virtual A { int b; };
struct C : B, virtual A { int c; };
C c; // Force vbtable emission.

// CHECK-DAG: @"??_8C@Test4@@7B@" = {{.*}} [2 x i32] [i32 0, i32 12]
// CHECK-DAG: @"??_8B@Test4@@7B@" = {{.*}} [2 x i32] [i32 0, i32 8]
}

namespace Test5 {
// Test multiple base subobjects of the same type when that type has a virtual
// base.
struct A { int a; };
struct B : virtual A { int b; };
struct C : B { int c; };
struct D : B, C { int d; };
D d; // Force vbtable emission.

// CHECK-DAG: @"??_8D@Test5@@7BB@1@@"
// CHECK-DAG: @"??_8D@Test5@@7BC@1@@"
// CHECK-DAG: @"??_8C@Test5@@7B@"
// CHECK-DAG: @"??_8B@Test5@@7B@"
}

namespace Test6 {
// Test that we skip unneeded base path component names.
struct A { int a; };
struct B : virtual A { int b; };
struct C : B { int c; };
struct D : B, C { int d; };
struct E : D { int e; };
struct F : E, B, C { int f; };
struct G : F, virtual E { int g; };
G g;

// CHECK-DAG: @"??_8G@Test6@@7BB@1@E@1@F@1@@" =
// CHECK-DAG: @"??_8G@Test6@@7BC@1@E@1@F@1@@" =
// CHECK-DAG: @"??_8G@Test6@@7BB@1@F@1@@" =
// CHECK-DAG: @"??_8G@Test6@@7BC@1@F@1@@" =
// CHECK-DAG: @"??_8G@Test6@@7BB@1@E@1@@" =
// CHECK-DAG: @"??_8G@Test6@@7BC@1@E@1@@" =
// CHECK-DAG: @"??_8F@Test6@@7BB@1@E@1@@" = {{.*}} [2 x i32] [i32 0, i32 52]
// CHECK-DAG: @"??_8F@Test6@@7BC@1@E@1@@" = {{.*}} [2 x i32] [i32 0, i32 44]
// CHECK-DAG: @"??_8F@Test6@@7BB@1@@" = {{.*}} [2 x i32] [i32 0, i32 24]
// CHECK-DAG: @"??_8F@Test6@@7BC@1@@" = {{.*}} [2 x i32] [i32 0, i32 16]
// CHECK-DAG: @"??_8C@Test6@@7B@" = {{.*}} [2 x i32] [i32 0, i32 12]
// CHECK-DAG: @"??_8B@Test6@@7B@" = {{.*}} [2 x i32] [i32 0, i32 8]
// CHECK-DAG: @"??_8E@Test6@@7BB@1@@" = {{.*}} [2 x i32] [i32 0, i32 28]
// CHECK-DAG: @"??_8E@Test6@@7BC@1@@" = {{.*}} [2 x i32] [i32 0, i32 20]
// CHECK-DAG: @"??_8D@Test6@@7BB@1@@" = {{.*}} [2 x i32] [i32 0, i32 24]
// CHECK-DAG: @"??_8D@Test6@@7BC@1@@" = {{.*}} [2 x i32] [i32 0, i32 16]
}

namespace Test7 {
// Test a non-virtual base which reuses the vbptr of another base.
struct A { int a; };
struct B { int b; };
struct C { int c; };
struct D : virtual A { int d; };
struct E : B, D, virtual A, virtual C { int e; };
E o;

// CHECK-DAG: @"??_8E@Test7@@7B@" = {{.*}} [3 x i32] [i32 0, i32 12, i32 16]
// CHECK-DAG: @"??_8D@Test7@@7B@" = {{.*}} [2 x i32] [i32 0, i32 8]
}

namespace Test8 {
// Test a virtual base which reuses the vbptr of another base.
struct A { int a; };
struct B : virtual A { int b; };
struct C : B { int c; };
struct D : virtual C { int d; };
D o;

// CHECK-DAG: @"??_8D@Test8@@7B01@@" = {{.*}} [3 x i32] [i32 0, i32 8, i32 12]
// CHECK-DAG: @"??_8D@Test8@@7BC@1@@" = {{.*}} [2 x i32] [i32 0, i32 -4]
// CHECK-DAG: @"??_8C@Test8@@7B@" = {{.*}} [2 x i32] [i32 0, i32 12]
// CHECK-DAG: @"??_8B@Test8@@7B@" = {{.*}} [2 x i32] [i32 0, i32 8]
}

namespace Test9 {
// D has to add to B's vbtable because D has more morally virtual bases than B.
// D then takes B's vbptr and the vbtable is named for D, not B.
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual B { int c; };
struct BB : B { int bb; };  // Indirection =/
struct D : BB, C { int d; };
struct E : virtual D { };
E e;

// CHECK-DAG: @"??_8E@Test9@@7B01@@" =
// CHECK-DAG: @"??_8E@Test9@@7BD@1@@" =
// CHECK-DAG: @"??_8E@Test9@@7BC@1@@" =
// CHECK-DAG: @"??_8E@Test9@@7BB@1@@" =
// CHECK-DAG: @"??_8D@Test9@@7B@" =
// CHECK-DAG: @"??_8D@Test9@@7BC@1@@" =
// CHECK-DAG: @"??_8D@Test9@@7BB@1@@" =
// CHECK-DAG: @"??_8C@Test9@@7B01@@" =
// CHECK-DAG: @"??_8C@Test9@@7BB@1@@" =
// CHECK-DAG: @"??_8BB@Test9@@7B@" =
// CHECK-DAG: @"??_8B@Test9@@7B@" =
}

namespace Test10 {
struct A { int a; };
struct B { int b; };
struct C : virtual A { int c; };
struct D : B, C { int d; };
D d;

// CHECK-DAG: @"??_8D@Test10@@7B@" =
// CHECK-DAG: @"??_8C@Test10@@7B@" =

}

namespace Test11 {
// Typical diamond with an extra single inheritance indirection for B and C.
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual A { int c; };
struct D : B { int d; };
struct E : C { int e; };
struct F : D, E { int f; };
F f;

// CHECK-DAG: @"??_8F@Test11@@7BD@1@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 28]
// CHECK-DAG: @"??_8F@Test11@@7BE@1@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 16]
// CHECK-DAG: @"??_8E@Test11@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 12]
// CHECK-DAG: @"??_8C@Test11@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 8]
// CHECK-DAG: @"??_8D@Test11@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 12]
// CHECK-DAG: @"??_8B@Test11@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 8]

}

namespace Test12 {
// Another vbptr inside a virtual base.
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual B { int c; };
struct D : C, B { int d; };
struct E : D, C, B { int e; };
E e;

// CHECK-DAG: @"??_8E@Test12@@7BC@1@D@1@@" =
// CHECK-DAG: @"??_8E@Test12@@7BB@1@D@1@@" =
// CHECK-DAG: @"??_8E@Test12@@7BD@1@@" =
// CHECK-DAG: @"??_8E@Test12@@7BC@1@@" =
// CHECK-DAG: @"??_8E@Test12@@7BB@1@@" =
// CHECK-DAG: @"??_8C@Test12@@7B01@@" =
// CHECK-DAG: @"??_8C@Test12@@7BB@1@@" =
// CHECK-DAG: @"??_8D@Test12@@7BC@1@@" =
// CHECK-DAG: @"??_8D@Test12@@7BB@1@@" =
// CHECK-DAG: @"??_8D@Test12@@7B@" =
// CHECK-DAG: @"??_8B@Test12@@7B@" =
}

namespace Test13 {
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual B { int c; };
struct D : virtual C { int d; };
struct E : D, C, B { int e; };
E e;

// CHECK-DAG: @"??_8E@Test13@@7BD@1@@" =
// CHECK-DAG: @"??_8E@Test13@@7BC@1@D@1@@" =
// CHECK-DAG: @"??_8E@Test13@@7BB@1@D@1@@" =
// CHECK-DAG: @"??_8E@Test13@@7BC@1@@" =
// CHECK-DAG: @"??_8E@Test13@@7BB@1@@" =
// CHECK-DAG: @"??_8D@Test13@@7B@" =
// CHECK-DAG: @"??_8D@Test13@@7BC@1@@" =
// CHECK-DAG: @"??_8D@Test13@@7BB@1@@" =
// CHECK-DAG: @"??_8C@Test13@@7B01@@" =
// CHECK-DAG: @"??_8C@Test13@@7BB@1@@" =
// CHECK-DAG: @"??_8B@Test13@@7B@" =
}

namespace Test14 {
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual B { int c; };
struct D : virtual C { int d; };
struct E : D, virtual C, virtual B { int e; };
E e;

// CHECK-DAG: @"??_8E@Test14@@7B@" =
// CHECK-DAG: @"??_8E@Test14@@7BC@1@@" =
// CHECK-DAG: @"??_8E@Test14@@7BB@1@@" =
// CHECK-DAG: @"??_8D@Test14@@7B@" =
// CHECK-DAG: @"??_8D@Test14@@7BC@1@@" =
// CHECK-DAG: @"??_8D@Test14@@7BB@1@@" =
// CHECK-DAG: @"??_8C@Test14@@7B01@@" =
// CHECK-DAG: @"??_8C@Test14@@7BB@1@@" =
// CHECK-DAG: @"??_8B@Test14@@7B@" =
}

namespace Test15 {
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual A { int c; };
struct D : virtual B { int d; };
struct E : D, C, B { int e; };
E e;

// CHECK-DAG: @"??_8E@Test15@@7BD@1@@" =
// CHECK-DAG: @"??_8E@Test15@@7BB@1@D@1@@" =
// CHECK-DAG: @"??_8E@Test15@@7BC@1@@" =
// CHECK-DAG: @"??_8E@Test15@@7BB@1@@" =
// CHECK-DAG: @"??_8C@Test15@@7B@" =
// CHECK-DAG: @"??_8D@Test15@@7B01@@" =
// CHECK-DAG: @"??_8D@Test15@@7BB@1@@" =
// CHECK-DAG: @"??_8B@Test15@@7B@" =
}

namespace Test16 {
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual B { int c; }; // ambig
struct D : virtual C { int d; };
struct E : virtual D { int e; }; // ambig
struct F : E, D, C, B { int f; };  // ambig
F f;

// CHECK-DAG: @"??_8F@Test16@@7BE@1@@" =
// CHECK-DAG: @"??_8F@Test16@@7BD@1@E@1@@" =
// CHECK-DAG: @"??_8F@Test16@@7BC@1@E@1@@" =
// CHECK-DAG: @"??_8F@Test16@@7BB@1@E@1@@" =
// CHECK-DAG: @"??_8F@Test16@@7BD@1@@" =
// CHECK-DAG: @"??_8F@Test16@@7BC@1@@" =
// CHECK-DAG: @"??_8F@Test16@@7BB@1@@" =
// CHECK-DAG: @"??_8E@Test16@@7B01@@" =
// CHECK-DAG: @"??_8E@Test16@@7BD@1@@" =
// CHECK-DAG: @"??_8E@Test16@@7BC@1@@" =
// CHECK-DAG: @"??_8E@Test16@@7BB@1@@" =
// CHECK-DAG: @"??_8D@Test16@@7B@" =
// CHECK-DAG: @"??_8D@Test16@@7BC@1@@" =
// CHECK-DAG: @"??_8D@Test16@@7BB@1@@" =
// CHECK-DAG: @"??_8C@Test16@@7B01@@" =
// CHECK-DAG: @"??_8C@Test16@@7BB@1@@" =
// CHECK-DAG: @"??_8B@Test16@@7B@" =
}

namespace Test17 {
// This test case has an interesting alternating pattern of using "vbtable of B"
// and "vbtable of C for C".  This may be the key to the underlying algorithm.
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual B { int c; }; // ambig
struct D : virtual C { int d; };
struct E : virtual D { int e; }; // ambig
struct F : virtual E { int f; };
struct G : virtual F { int g; }; // ambig
struct H : virtual G { int h; };
struct I : virtual H { int i; }; // ambig
struct J : virtual I { int j; };
struct K : virtual J { int k; }; // ambig
K k;

// CHECK-DAG: @"??_8K@Test17@@7B01@@" =
// CHECK-DAG: @"??_8J@Test17@@7B@" =
// CHECK-DAG: @"??_8I@Test17@@7B01@@" =
// CHECK-DAG: @"??_8H@Test17@@7B@" =
// CHECK-DAG: @"??_8G@Test17@@7B01@@" =
// CHECK-DAG: @"??_8F@Test17@@7B@" =
// CHECK-DAG: @"??_8E@Test17@@7B01@@" =
// CHECK-DAG: @"??_8D@Test17@@7B@" =
// CHECK-DAG: @"??_8C@Test17@@7B01@@" =
// CHECK-DAG: @"??_8B@Test17@@7B@" =
}

namespace Test18 {
struct A { int a; };
struct B : virtual A { int b; };
struct C : B { int c; };
struct D : C, B { int d; };
struct E : D, C, B { int e; };
E e;

// CHECK-DAG: @"??_8E@Test18@@7BC@1@D@1@@" =
// CHECK-DAG: @"??_8E@Test18@@7BB@1@D@1@@" =
// CHECK-DAG: @"??_8E@Test18@@7BC@1@@" =
// CHECK-DAG: @"??_8E@Test18@@7BB@1@@" =
// CHECK-DAG: @"??_8B@Test18@@7B@" =
// CHECK-DAG: @"??_8C@Test18@@7B@" =
// CHECK-DAG: @"??_8D@Test18@@7BC@1@@" =
// CHECK-DAG: @"??_8D@Test18@@7BB@1@@" =
}

namespace Test19 {
struct A { int a; };
struct B : virtual A { int b; };
struct C : virtual B { int c; };
struct D : virtual C, virtual B { int d; };
struct E : virtual D, virtual C, virtual B { int e; };
E e;

// CHECK-DAG: @"??_8E@Test19@@7B01@@" =
// CHECK-DAG: @"??_8E@Test19@@7BD@1@@" =
// CHECK-DAG: @"??_8E@Test19@@7BC@1@@" =
// CHECK-DAG: @"??_8E@Test19@@7BB@1@@" =
// CHECK-DAG: @"??_8D@Test19@@7B@" =
// CHECK-DAG: @"??_8D@Test19@@7BC@1@@" =
// CHECK-DAG: @"??_8D@Test19@@7BB@1@@" =
// CHECK-DAG: @"??_8C@Test19@@7B01@@" =
// CHECK-DAG: @"??_8C@Test19@@7BB@1@@" =
// CHECK-DAG: @"??_8B@Test19@@7B@" =
}

namespace Test20 {
// E has no direct vbases, but it adds to C's vbtable anyway.
struct A { int a; };
struct B { int b; };
struct C : virtual A { int c; };
struct D : virtual B { int d; };
struct E : C, D { int e; };
E f;

// CHECK-DAG: @"??_8E@Test20@@7BC@1@@" = linkonce_odr unnamed_addr constant [3 x i32] [i32 0, i32 20, i32 24]
// CHECK-DAG: @"??_8E@Test20@@7BD@1@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 16]
// CHECK-DAG: @"??_8D@Test20@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 8]
// CHECK-DAG: @"??_8C@Test20@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 8]
}

namespace Test21 {
struct A { int a; };
struct B : virtual A { int b; };
struct C : B { int c; };
struct D : B { int d; };
struct E : C, D { int e; };
struct F : virtual E { int f; };
struct G : E { int g; };
struct H : F, G { int h; };
H h;

// CHECK-DAG: @"??_8H@Test21@@7B@" =
// CHECK-DAG: @"??_8H@Test21@@7BC@1@F@1@@" =
// CHECK-DAG: @"??_8H@Test21@@7BD@1@F@1@@" =
// CHECK-DAG: @"??_8H@Test21@@7BC@1@G@1@@" =
// CHECK-DAG: @"??_8H@Test21@@7BD@1@G@1@@" =
// CHECK-DAG: @"??_8G@Test21@@7BC@1@@" =
// CHECK-DAG: @"??_8G@Test21@@7BD@1@@" =
// CHECK-DAG: @"??_8F@Test21@@7B@" =
// CHECK-DAG: @"??_8F@Test21@@7BC@1@@" =
// CHECK-DAG: @"??_8F@Test21@@7BD@1@@" =
// CHECK-DAG: @"??_8E@Test21@@7BC@1@@" =
// CHECK-DAG: @"??_8E@Test21@@7BD@1@@" =
// CHECK-DAG: @"??_8D@Test21@@7B@" =
// CHECK-DAG: @"??_8B@Test21@@7B@" =
// CHECK-DAG: @"??_8C@Test21@@7B@" =
}

namespace Test22 {
struct A { int a; };
struct B : virtual A { int b; };
struct C { int c; };
struct D : B, virtual C { int d; };
D d;

// CHECK-DAG: @"??_8D@Test22@@7B@" = linkonce_odr unnamed_addr constant [3 x i32] [i32 0, i32 12, i32 16]
// CHECK-DAG: @"??_8B@Test22@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 8]
}

namespace Test23 {
struct A { int a; };
struct B : virtual A { int b; };
struct C { int c; };
// Note the unusual order of bases. It forces C to be laid out before A.
struct D : virtual C, B { int d; };
D d;

// CHECK-DAG: @"??_8D@Test23@@7B@" = linkonce_odr unnamed_addr constant [3 x i32] [i32 0, i32 16, i32 12]
// CHECK-DAG: @"??_8B@Test23@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 8]
}

namespace Test24 {
struct A { int a; };
struct B : virtual A { int b; };
struct C { int c; };
struct D : virtual C, B {
  virtual void f();  // Issues a vfptr, but the vbptr is still shared with B.
  int d;
};
D d;

// CHECK-DAG: @"??_8D@Test24@@7B@" = linkonce_odr unnamed_addr constant [3 x i32] [i32 0, i32 16, i32 12]
// CHECK-DAG: @"??_8B@Test24@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 8]
}

namespace Test25 {
struct A { int a; };
struct B : virtual A {
  virtual void f();  // Issues a vfptr.
  int b;
};
struct C { int c; };
struct D : virtual C, B { int d; };
D d;

// CHECK-DAG: @"??_8D@Test25@@7B@" = linkonce_odr unnamed_addr constant [3 x i32] [i32 -4, i32 16, i32 12]
// CHECK-DAG: @"??_8B@Test25@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 -4, i32 8]
}

namespace Test26 {
struct A { int a; };
struct B { int b; };
struct C { int c; };
struct D : virtual A { int d; };
struct E : virtual B {
  virtual void foo();  // Issues a vfptr.
  int e;
};
struct F: virtual C, D, E { int f; };
F f;
// F reuses the D's vbptr, even though D is laid out after E.
// CHECK-DAG: @"??_8F@Test26@@7BD@1@@" = linkonce_odr unnamed_addr constant [4 x i32] [i32 0, i32 16, i32 12, i32 20]
// CHECK-DAG: @"??_8F@Test26@@7BE@1@@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 -4, i32 28]
}

namespace Test27 {
// PR17748
struct A {};
struct B : virtual A {};
struct C : virtual B {};
struct D : C, B {};
struct E : D {};
struct F : C, E {};
struct G : F, D, C, B {};
G x;

// CHECK-DAG: @"??_8G@Test27@@7BB@1@@" =
// CHECK-DAG: @"??_8G@Test27@@7BB@1@F@1@@" =
// CHECK-DAG: @"??_8G@Test27@@7BC@1@@" =
// CHECK-DAG: @"??_8G@Test27@@7BC@1@D@1@@" =
// CHECK-DAG: @"??_8G@Test27@@7BC@1@E@1@@" =
// CHECK-DAG: @"??_8G@Test27@@7BC@1@F@1@@" =
// CHECK-DAG: @"??_8G@Test27@@7BD@1@@" =
// CHECK-DAG: @"??_8G@Test27@@7BF@1@@" =
}

namespace Test28 {
// PR17748
struct A {};
struct B : virtual A {};
struct C : virtual B {};
struct D : C, B {};
struct E : C, D {};
struct F : virtual E, virtual D, virtual C {};
F x;

// CHECK-DAG: @"??_8F@Test28@@7B01@@" =
// CHECK-DAG: @"??_8F@Test28@@7BB@1@@" =
// CHECK-DAG: @"??_8F@Test28@@7BC@1@@" =
// CHECK-DAG: @"??_8F@Test28@@7BC@1@D@1@@" =
// CHECK-DAG: @"??_8F@Test28@@7BC@1@D@1@E@1@@" =
// CHECK-DAG: @"??_8F@Test28@@7BC@1@E@1@@" =
// CHECK-DAG: @"??_8F@Test28@@7BD@1@@" =
// CHECK-DAG: @"??_8F@Test28@@7BE@1@@" =
}

namespace Test29 {
struct A {};
struct B : virtual A {};
struct C : virtual B {};
struct D : C {};
D d;

// CHECK-DAG: @"??_8D@Test29@@7BB@1@@" = linkonce_odr unnamed_addr constant [2 x i32] zeroinitializer
}

namespace Test30 {
struct A {};
template <class> struct B : virtual A {
  B() {}
};

extern template class B<int>;
template B<int>::B();
// CHECK-DAG: @"??_8?$B@H@Test30@@7B@" = linkonce_odr unnamed_addr constant [2 x i32] [i32 0, i32 4], comdat
}
