// RUN: %clang_cc1 %s -Wno-uninitialized -std=c++17 -fsyntax-only -verify

namespace Vector {

using TwoIntsVecSize __attribute__((vector_size(8))) = int;

constexpr TwoIntsVecSize a = {1,2};
static_assert(a[1] == 2);
static_assert(a[2]); // expected-error {{not an integral constant expression}} expected-note {{read of dereferenced one-past-the-end pointer}}

}

namespace ExtVector {

using FourIntsExtVec __attribute__((ext_vector_type(4))) = int;

constexpr FourIntsExtVec b = {1,2,3,4};
static_assert(b[0] == 1 && b[1] == 2 && b[2] == 3 && b[3] == 4);
static_assert(b.s0 == 1 && b.s1 == 2 && b.s2 == 3 && b.s3 == 4);
static_assert(b.x == 1 && b.y == 2 && b.z == 3 && b.w == 4);
static_assert(b.r == 1 && b.g == 2 && b.b == 3 && b.a == 4);
static_assert(b[5]); // expected-error {{not an integral constant expression}} expected-note {{read of dereferenced one-past-the-end pointer}}

// FIXME: support selecting multiple elements
static_assert(b.lo.lo == 1); // expected-error {{not an integral constant expression}}
// static_assert(b.lo.lo==1 && b.lo.hi==2 && b.hi.lo == 3 && b.hi.hi == 4);
// static_assert(b.odd[0]==1 && b.odd[1]==2 && b.even[0] == 3 && b.even[1] == 4);

}
