// RUN: %clang_cc1 -std=gnu++2a -emit-pch %s -o %t.pch
// RUN: %clang_cc1 -std=gnu++2a %s -DEMIT -ast-merge %t.pch -ast-dump-all | FileCheck %s

// XFAIL: *

#ifndef EMIT
#define EMIT

namespace Integer {

consteval int fint() {
  return 6789;
}

int Unique_Int = fint();
//CHECK:      VarDecl {{.*}} Unique_Int
//CHECK-NEXT: ConstantExpr {{.*}} 'int'
//CHECK-NEXT: value: Int 6789

consteval __uint128_t fint128() {
  return ((__uint128_t)0x75f17d6b3588f843 << 64) | 0xb13dea7c9c324e51;
}

constexpr __uint128_t Unique_Int128 = fint128();
//CHECK:      VarDecl {{.*}} Unique_Int128
//CHECK-NEXT: value: Int 156773562844924187900898496343692168785
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: Int 156773562844924187900898496343692168785

} // namespace Integer

namespace FloatingPoint {

consteval double fdouble() {
  return double(567890.67890);
}

double Unique_Double = fdouble();
//CHECK:      VarDecl {{.*}} Unique_Double
//CHECK-NEXT: ConstantExpr {{.*}}
//CHECK-NEXT: value: Float 5.678907e+05

} // namespace FloatingPoint

// FIXME: Add test for FixedPoint, ComplexInt, ComplexFloat, AddrLabelDiff.

namespace Struct {

struct B {
  int i;
  double d;
};

consteval B fB() {
  return B{1, 0.7};
}

constexpr B Basic_Struct = fB();
//CHECK:      VarDecl {{.*}} Basic_Struct
//CHECK-NEXT: value: Struct
//CHECK-NEXT: fields: Int 1, Float 7.000000e-01
//CHECK-NEXT: ImplicitCastExpr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: Struct
//CHECK-NEXT: fields: Int 1, Float 7.000000e-01

struct C {
  int i = 9;
};

struct A : B {
  constexpr A(B b, int I, double D, C _c) : B(b), i(I), d(D), c(_c) {}
  int i;
  double d;
  C c;
};

consteval A fA() {
  return A(Basic_Struct, 1, 79.789, {});
}

A Advanced_Struct = fA();
//CHECK:      VarDecl {{.*}} Advanced_Struct
//CHECK-NEXT: ConstantExpr {{.*}}
//CHECK-NEXT: value: Struct
//CHECK-NEXT: base: Struct
//CHECK-NEXT: fields: Int 1, Float 7.000000e-01
//CHECK-NEXT: fields: Int 1, Float 7.978900e+01
//CHECK-NEXT: field: Struct
//CHECK-NEXT: field: Int 9

} // namespace Struct

namespace Vector {

using v4si = int __attribute__((__vector_size__(16)));

consteval v4si fv4si() {
  return (v4si){8, 2, 3};
}

v4si Vector_Int = fv4si();
//CHECK:      VarDecl {{.*}} Vector_Int
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: Vector length=4
//CHECK-NEXT: elements: Int 8, Int 2, Int 3, Int 0

} // namespace Vector

namespace Array {

struct B {
  int arr[6];
};

consteval B fint() {
  return B{1, 2, 3, 4, 5, 6};
}

B Array_Int = fint();
//CHECK:      VarDecl {{.*}} Array_Int
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: Struct
//CHECK-NEXT: field: Array size=6
//CHECK-NEXT: elements: Int 1, Int 2, Int 3, Int 4
//CHECK-NEXT: elements: Int 5, Int 6

struct A {
  int i = 789;
  double d = 67890.09876;
};

struct C {
  A arr[3];
};

consteval C fA() {
  return {{A{}, A{-45678, 9.8}, A{9}}};
}

C Array2_Struct = fA();
//CHECK:      VarDecl {{.*}} Array2_Struct
//CHECK-NEXT: ConstantExpr {{.*}}

using v4si = int __attribute__((__vector_size__(16)));

struct D {
  v4si arr[2];
};

consteval D fv4si() {
  return {{{1, 2, 3, 4}, {4, 5, 6, 7}}};
}

D Array_Vector = fv4si();
//CHECK:      VarDecl {{.*}} Array_Vector
//CHECK-NEXT: ConstantExpr {{.*}}
//CHECK-NEXT: value: Struct
//CHECK-NEXT: field: Array size=2
//CHECK-NEXT: element: Vector length=4
//CHECK-NEXT: elements: Int 1, Int 2, Int 3, Int 4
//CHECK-NEXT: element: Vector length=4
//CHECK-NEXT: elements: Int 4, Int 5, Int 6, Int 7

} // namespace Array

namespace Union {

struct A {
  int i = 6789;
  float f = 987.9876;
};

union U {
  int i;
  A a{567890, 9876.5678f};
};

consteval U fU1() {
  return U{0};
}

U Unique_Union1 = fU1();
//CHECK:      VarDecl {{.*}} Unique_Union
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: Union .i Int 0

consteval U fU() {
  return U{};
}

U Unique_Union2 = fU();
//CHECK:      VarDecl {{.*}} Unique_Union
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: Union .a
//CHECK-NEXT: Struct
//CHECK-NEXT: fields: Int 567890, Float 9.876567e+03

} // namespace Union

namespace MemberPointer {

struct A {
  struct B {
    struct C {
      struct D {
        struct E {
          struct F {
            struct G {
              int i;
            };
          };
        };
      };
    };
  };
};

consteval auto fmem_ptr() -> decltype(&A::B::C::D::E::F::G::i) {
  return &A::B::C::D::E::F::G::i;
}

auto MemberPointer1 = fmem_ptr();
//CHECK:      VarDecl {{.*}} MemberPointer1
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: MemberPointer &G::i

struct A1 {
  struct B1 {
    int f() const {
      return 0;
    }
  };
};

consteval auto fmem_ptr2() {
  return &A1::B1::f;
}

auto MemberPointer2 = fmem_ptr2();
//CHECK:      VarDecl {{.*}} MemberPointer2
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: MemberPointer &B1::f

} // namespace MemberPointer

namespace std {
struct type_info;
};

namespace LValue {

constexpr int g = 0;

consteval const int &fg_ref() {
  return g;
}

const int &g_ref = fg_ref();
//CHECK:      VarDecl {{.*}} g_ref
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: LValue &g

consteval const int *fint_ptr() {
  return &g;
}

const int *g_ptr = fint_ptr();
//CHECK:      VarDecl {{.*}} g_ptr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: LValue &g

consteval const int *fnull_ptr() {
  return nullptr;
}

const int *ptr2 = fnull_ptr();
//CHECK:      VarDecl {{.*}} ptr2
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: LValue nullptr

int fconst();

consteval auto ffunc_ptr() {
  return &fconst;
}

int (*func_ptr)() = ffunc_ptr();
//CHECK:      VarDecl {{.*}} func_ptr
//CHECK-NEXT: ConstantExpr {{.*}}
//CHECK-NEXT: value: LValue &fconst

struct A {
  int Arr[6] = {0, 1, 3, 4, 5, 9};
  int i = 0;
};

struct D {
  A arr[6] = {};
};

consteval D fA() {
  return {};
}

constexpr D Arr = fA();
// CHECK:      VarDecl {{.*}} Arr
// CHECK-NEXT: value: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0
// CHECK-NEXT: element: Struct
// CHECK-NEXT: field: Array size=6
// CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
// CHECK-NEXT: elements: Int 5, Int 9
// CHECK-NEXT: field: Int 0

consteval const int &fconstintref() {
  return Arr.arr[0].i;
}

const int &ArrayStructRef1 = fconstintref();
//CHECK:      VarDecl {{.*}} ArrayStructRef1
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: LValue &Arr.arr[0].i

consteval const int &fconstintref2() {
  return Arr.arr[1].Arr[5];
}

const int &ArrayStructRef2 = fconstintref2();
//CHECK:      VarDecl {{.*}} ArrayStructRef2
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: LValue &Arr.arr[1].Arr[5]

consteval const int *fconststar() {
  return &ArrayStructRef2;
}

const int *ArrayStructRef3 = fconststar();
//CHECK:      VarDecl {{.*}} ArrayStructRef3
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: LValue  &Arr.arr[1].Arr[5]

struct B : A {
};

struct C {
  B b;
};

consteval C fC() {
  return {};
}

C c = fC();
//CHECK:      VarDecl {{.*}} c
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: Struct
//CHECK-NEXT: field: Struct
//CHECK-NEXT: base: Struct
//CHECK-NEXT: field: Array size=6
//CHECK-NEXT: elements: Int 0, Int 1, Int 3, Int 4
//CHECK-NEXT: elements: Int 5, Int 9
//CHECK-NEXT: field: Int 0

consteval const int &f2constintref() {
  return c.b.i;
}

const int &StructPathRef = f2constintref();
//CHECK:      VarDecl {{.*}} StructPathRef
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: LValue &c.b.A::i

consteval const std::type_info *ftype_info() {
  return &typeid(c);
}

const std::type_info *T1 = ftype_info();
//CHECK:      VarDecl {{.*}} T1
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT:value: LValue &typeid(LValue::C)

consteval const std::type_info *ftype_info2() {
  return &typeid(Arr.arr[1].Arr[2]);
}

const std::type_info *T2 = ftype_info2();
//CHECK:      VarDecl {{.*}} T2
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: LValue &typeid(int)

consteval const char *fstring() {
  return "test";
}

const char *cptr = fstring();
//CHECK:      VarDecl {{.*}} cptr
//CHECK-NEXT: ConstantExpr
//CHECK-NEXT: value: LValue &"test"[0]

} // namespace LValue

#endif
