// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++20 -x objective-c++ -fobjc-arc -fenable-matrix -triple i686-pc-win32

enum class N {};

using B1 = int;
using X1 = B1;
using Y1 = B1;

using B2 = void;
using X2 = B2;
using Y2 = B2;

using A3 = char __attribute__((vector_size(4)));
using B3 = A3;
using X3 = B3;
using Y3 = B3;

using A4 = float;
using B4 = A4 __attribute__((matrix_type(4, 4)));
using X4 = B4;
using Y4 = B4;

using X5 = A4 __attribute__((matrix_type(3, 4)));
using Y5 = A4 __attribute__((matrix_type(4, 3)));

N t1 = 0 ? X1() : Y1(); // expected-error {{rvalue of type 'B1'}}
N t2 = 0 ? X2() : Y2(); // expected-error {{rvalue of type 'B2'}}

const X1 &xt3 = 0;
const Y1 &yt3 = 0;
N t3 = 0 ? xt3 : yt3; // expected-error {{lvalue of type 'const B1'}}

N t4 = X3() + Y3();   // expected-error {{rvalue of type 'B3'}}

N t5 = A3() ? X3() : Y3(); // expected-error {{rvalue of type 'B3'}}
N t6 = A3() ? X1() : Y1(); // expected-error {{vector condition type 'A3' (vector of 4 'char' values) and result type '__attribute__((__vector_size__(4 * sizeof(B1)))) B1' (vector of 4 'B1' values) do not have elements of the same size}}

N t7 = X4() + Y4(); // expected-error {{rvalue of type 'B4'}}
N t8 = X4() * Y4(); // expected-error {{rvalue of type 'B4'}}
N t9 = X5() * Y5(); // expected-error {{rvalue of type 'A4 __attribute__((matrix_type(3, 3)))'}}

template <class T> struct S1 {
  template <class U> struct S2 {};
};

N t10 = 0 ? S1<X1>() : S1<Y1>(); // expected-error {{from 'S1<B1>' (aka 'S1<int>')}}
// FIXME: needs to compute common sugar for qualified template names
N t11 = 0 ? S1<X1>::S2<X2>() : S1<Y1>::S2<Y2>(); // expected-error {{from 'S1<int>::S2<B2>' (aka 'S1<int>::S2<void>')}}

template <class T> using Al = S1<T>;

N t12 = 0 ? Al<X1>() : Al<Y1>(); // expected-error {{from 'Al<B1>' (aka 'S1<int>')}}

#define AS1 __attribute__((address_space(1)))
#define AS2 __attribute__((address_space(1)))
using AS1X1 = AS1 B1;
using AS1Y1 = AS1 B1;
using AS2Y1 = AS2 B1;
N t13 = 0 ? (AS1X1){} : (AS1Y1){}; // expected-error {{rvalue of type 'AS1 B1' (aka '__attribute__((address_space(1))) int')}}
N t14 = 0 ? (AS1X1){} : (AS2Y1){}; // expected-error {{rvalue of type '__attribute__((address_space(1))) B1' (aka '__attribute__((address_space(1))) int')}}

using FX1 = X1 ();
using FY1 = Y1 ();
N t15 = 0 ? (FX1*){} : (FY1*){}; // expected-error {{rvalue of type 'B1 (*)()' (aka 'int (*)()')}}

struct SS1 {};
using SB1 = SS1;
using SX1 = SB1;
using SY1 = SB1;

using MFX1 = X1 SX1::*();
using MFY1 = Y1 SY1::*();

N t16 = 0 ? (MFX1*){} : (MFY1*){}; // expected-error {{rvalue of type 'B1 SB1::*(*)()'}}

N t17 = 0 ? (FX1 SX1::*){} : (FY1 SY1::*){}; // expected-error {{rvalue of type 'B1 (SB1::*)() __attribute__((thiscall))'}}

N t18 = 0 ? (__typeof(X1*)){} : (__typeof(Y1*)){}; // expected-error {{rvalue of type 'typeof(B1 *)' (aka 'int *')}}

struct Enums {
  enum X : B1;
  enum Y : ::B1;
};
using EnumsB = Enums;
using EnumsX = EnumsB;
using EnumsY = EnumsB;

N t19 = 0 ? (__underlying_type(EnumsX::X)){} : (__underlying_type(EnumsY::Y)){};
// expected-error@-1 {{rvalue of type 'B1' (aka 'int')}}

N t20 = 0 ? (__underlying_type(EnumsX::X)){} : (__underlying_type(EnumsY::X)){};
// expected-error@-1 {{rvalue of type '__underlying_type(EnumsB::X)' (aka 'int')}}

using QX = const SB1 *;
using QY = const ::SB1 *;
N t23 = 0 ? (QX){} : (QY){}; // expected-error {{rvalue of type 'const SB1 *' (aka 'const SS1 *')}}

template <class T> using Alias = short;
N t24 = 0 ? (Alias<X1>){} : (Alias<Y1>){}; // expected-error {{rvalue of type 'Alias<B1>' (aka 'short')}}
N t25 = 0 ? (Alias<X1>){} : (Alias<X2>){}; // expected-error {{rvalue of type 'short'}}

template <class T, class U> concept C1 = true;
template <class T, class U> concept C2 = true;
C1<X1> auto t26_1 = (SB1){};
C1<X2> auto t26_2 = (::SB1){};
C2<X2> auto t26_3 = (::SB1){};
N t26 = 0 ? t26_1 : t26_2; // expected-error {{from 'SB1' (aka 'SS1')}}
N t27 = 0 ? t26_1 : t26_3; // expected-error {{from 'SB1' (aka 'SS1')}}

using RPB1 = X1*;
using RPX1 = RPB1;
using RPB1 = Y1*; // redeclared
using RPY1 = RPB1;
N t28 = *(RPB1){}; // expected-error {{lvalue of type 'Y1' (aka 'int')}}
auto t29 = 0 ? (RPX1){} : (RPY1){};
N t30 = t29;  // expected-error {{lvalue of type 'RPB1' (aka 'int *')}}
N t31 = *t29; // expected-error {{lvalue of type 'B1' (aka 'int')}}

namespace A { using type1 = X1*; };
namespace C { using A::type1; };
using UPX1 = C::type1;
namespace A { using type1 = Y1*; };  // redeclared
namespace C { using A::type1; };     // redeclared
using UPY1 = C::type1;
auto t32 = 0 ? (UPX1){} : (UPY1){};
N t33 = t32;  // expected-error {{lvalue of type 'C::type1' (aka 'int *')}}
N t34 = *t32; // expected-error {{lvalue of type 'B1' (aka 'int')}}

// See https://github.com/llvm/llvm-project/issues/61419
namespace PR61419 {
  template <class T0, class T1> struct pair {
    T0 first;
    T1 second;
  };

  extern const pair<id, id> p;
  id t = false ? p.first : p.second;
} // namespace PR61419

namespace GH67603 {
  template <class> using A = long;
  template <class B> void h() {
    using C = B;
    using D = B;
    N t = 0 ? A<decltype(C())>() : A<decltype(D())>();
    // expected-error@-1 {{rvalue of type 'A<decltype(C())>' (aka 'long')}}
  }
  template void h<int>();
} // namespace GH67603

namespace arrays {
  namespace same_canonical {
    using ConstB1I = const B1[];
    using ConstB1C = const B1[1];
    const ConstB1I a = {0};
    const ConstB1C b = {0};
    N ta = a;
    // expected-error@-1 {{lvalue of type 'const B1[1]' (aka 'const int[1]')}}
    N tb = b;
    // expected-error@-1 {{lvalue of type 'const ConstB1C' (aka 'const const int[1]')}}
    N tc = 0 ? a : b;
    // expected-error@-1 {{lvalue of type 'const B1[1]' (aka 'const int[1]')}}
  } // namespace same_canonical
  namespace same_element {
    using ConstB1 = const B1;
    using ConstB1I = ConstB1[];
    using ConstB1C = ConstB1[1];
    const ConstB1I a = {0};
    const ConstB1C b = {0};
    N ta = a;
    // expected-error@-1 {{lvalue of type 'const ConstB1[1]' (aka 'const int[1]')}}
    N tb = b;
    // expected-error@-1 {{lvalue of type 'const ConstB1C' (aka 'const const int[1]')}}
    N tc = 0 ? a : b;
    // expected-error@-1 {{lvalue of type 'ConstB1[1]' (aka 'const int[1]')}}
  } // namespace same_element
  namespace balanced_qualifiers {
    using ConstX1C = const volatile X1[1];
    using Y1C = volatile Y1[1];
    extern volatile ConstX1C a;
    extern const volatile Y1C b;
    N ta = a;
    // expected-error@-1 {{lvalue of type 'volatile ConstX1C' (aka 'volatile const volatile int[1]')}}
    N tb = b;
    // expected-error@-1 {{lvalue of type 'const volatile Y1C' (aka 'const volatile volatile int[1]')}}
    N tc = 0 ? a : b;
    // expected-error@-1 {{lvalue of type 'const volatile volatile B1[1]' (aka 'const volatile volatile int[1]')}}
  } // namespace balanced_qualifiers
} // namespace arrays

namespace member_pointers {
  template <class T> struct W {
    X1 a;
    Y1 b;
  };
  struct W1 : W<X2> {};
  struct W2 : W<Y2> {};

  N t1 = 0 ? &W<X2>::a : &W<Y2>::b;
  // expected-error@-1 {{rvalue of type 'B1 W<B2>::*'}}

  // FIXME: adjusted MemberPointer does not preserve qualifier
  N t3 = 0 ? &W1::a : &W2::b;
  // expected-error@-1 {{rvalue of type 'B1 member_pointers::W<void>::*'}}
} // namespace member_pointers

namespace FunctionTypeExtInfo {
  namespace RecordType {
    class A;
    void (*x)(__attribute__((swift_async_context)) A *);

    class A;
    void (*y)(__attribute__((swift_async_context)) A *);

    N t1 = 0 ? x : y;
    // expected-error@-1 {{lvalue of type 'void (*)(__attribute__((swift_async_context)) A *)'}}
  } // namespace RecordType
  namespace TypedefType {
    class A;
    using B = A;
    void (*x)(__attribute__((swift_async_context)) B *);

    using B = A;
    void (*y)(__attribute__((swift_async_context)) B *);

    N t1 = 0 ? x : y;
    // expected-error@-1 {{lvalue of type 'void (*)(__attribute__((swift_async_context)) B *)'}}
  } // namespace TypedefType
} // namespace FunctionTypeExtInfo
