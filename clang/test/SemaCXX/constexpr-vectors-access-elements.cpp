// RUN: %clang_cc1 %s -Wno-uninitialized -std=c++17 -fsyntax-only -verify=expected,cpp17
// RUN: %clang_cc1 %s -Wno-uninitialized -std=c++20 -fsyntax-only -verify=expected,cpp20

namespace Vector {

using TwoIntsVecSize __attribute__((vector_size(8))) = int;

constexpr TwoIntsVecSize a = {1,2};
static_assert(a[1] == 2);
static_assert(a[2]); // expected-error {{not an integral constant expression}} expected-note {{read of dereferenced one-past-the-end pointer}}

constexpr struct {
    TwoIntsVecSize b;
} Val = {{0,1}};

static_assert(Val.b[1] == 1);

constexpr TwoIntsVecSize c[3] = {{0,1}, {2,3}, {4,5}};
static_assert(c[0][0] == 0);
static_assert(c[1][1] == 3);
static_assert(c[2][3]); // expected-error {{not an integral constant expression}} expected-note {{cannot refer to element 3 of array of 2 elements}}

// make sure clang rejects taking address of a vector element
static_assert(&a[0]); // expected-error {{address of vector element requested}}

}

namespace ExtVector {

using FourIntsExtVec __attribute__((ext_vector_type(4))) = int;

constexpr FourIntsExtVec b = {1,2,3,4};
static_assert(b[0] == 1 && b[1] == 2 && b[2] == 3 && b[3] == 4);
static_assert(b.s0 == 1 && b.s1 == 2 && b.s2 == 3 && b.s3 == 4);
static_assert(b.x == 1 && b.y == 2 && b.z == 3 && b.w == 4);
static_assert(b.r == 1 && b.g == 2 && b.b == 3 && b.a == 4);
static_assert(b[5]); // expected-error {{not an integral constant expression}} expected-note {{cannot refer to element 5 of array of 4 elements}}

// FIXME: support selecting multiple elements
static_assert(b.lo.lo == 1); // expected-error {{not an integral constant expression}}
// static_assert(b.lo.lo==1 && b.lo.hi==2 && b.hi.lo == 3 && b.hi.hi == 4);
// static_assert(b.odd[0]==1 && b.odd[1]==2 && b.even[0] == 3 && b.even[1] == 4);

// make sure clang rejects taking address of a vector element
static_assert(&b[1]); // expected-error {{address of vector element requested}}

constexpr const FourIntsExtVec *p = &b;
static_assert(p->x == 1);
}

namespace GH180044 {
template <typename T> constexpr T test1(char c) {
  T v;
  for (int i = 0; i < sizeof(T); ++i)
    v[i] = c;
  return v;
}

using C = char __attribute__((vector_size(16)));
C t1 = test1<C>(~1);

constexpr C t2 = test1<C>(~1);
static_assert(t2[0] == -2);
static_assert(t2[15] == -2);

using I = int __attribute__((vector_size(16)));

// expected-error@+1 {{constexpr function never produces a constant expression}}
constexpr unsigned test2() {
  // cpp17-warning@+1 {{uninitialized variable in a constexpr function is a C++20 extension}}
  I v;

  // expected-note@+2 {{subexpression not valid in a constant expression}}
  // expected-note@+1 {{subexpression not valid in a constant expression}}
  return __builtin_bit_cast(unsigned, v[0]);
}

// expected-error@+2 {{static assertion expression is not an integral constant expression}}
// expected-note@+1 {{in call to 'test2()'}}
static_assert(test2(), "");
}
