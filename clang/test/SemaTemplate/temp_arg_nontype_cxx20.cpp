// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// floating-point arguments
template<float> struct Float {};
using F1 = Float<1.0f>; // FIXME expected-error {{sorry}}
using F1 = Float<2.0f / 2>; // FIXME expected-error {{sorry}}

struct S { int n[3]; } s; // expected-note 1+{{here}}
int n; // expected-note 1+{{here}}

// pointers to subobjects
template<int *> struct IntPtr {};
using IPn = IntPtr<&n + 1>; // FIXME expected-error {{refers to subobject}}
using IPn = IntPtr<&n + 1>; // FIXME expected-error {{refers to subobject}}

using IP2 = IntPtr<&s.n[2]>; // FIXME expected-error {{refers to subobject}}
using IP2 = IntPtr<s.n + 2>; // FIXME expected-error {{refers to subobject}}

using IP3 = IntPtr<&s.n[3]>; // FIXME expected-error {{refers to subobject}}
using IP3 = IntPtr<s.n + 3>; // FIXME expected-error {{refers to subobject}}

template<int &> struct IntRef {};
using IPn = IntRef<*(&n + 1)>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of 'n'}}
using IPn = IntRef<*(&n + 1)>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of 'n'}}

using IP2 = IntRef<s.n[2]>; // FIXME expected-error {{refers to subobject}}
using IP2 = IntRef<*(s.n + 2)>; // FIXME expected-error {{refers to subobject}}

using IP3 = IntRef<s.n[3]>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of subobject of 's'}}
using IP3 = IntRef<*(s.n + 3)>; // expected-error {{not a constant expression}} expected-note {{dereferenced pointer past the end of subobject of 's'}}

// classes
template<S> struct Struct {};
using S123 = Struct<S{1, 2, 3}>; // FIXME: expected-error {{sorry}}
using S123 = Struct<S{1, 2, 3}>; // FIXME: expected-error {{sorry}}

// miscellaneous scalar types
template<_Complex int> struct ComplexInt {};
using CI = ComplexInt<1 + 3i>; // FIXME: expected-error {{sorry}}
using CI = ComplexInt<1 + 3i>; // FIXME: expected-error {{sorry}}

template<_Complex float> struct ComplexFloat {};
using CF = ComplexFloat<1.0f + 3.0fi>; // FIXME: expected-error {{sorry}}
using CF = ComplexFloat<1.0f + 3.0fi>; // FIXME: expected-error {{sorry}}
