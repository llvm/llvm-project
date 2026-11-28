// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -std=hlsl2021 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -std=hlsl202x -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -std=hlsl202y -fsyntax-only -verify %s

#if __HLSL_VERSION < 2028
// expected-warning@*{{support for HLSL language version hlsl2021 is incomplete, recommend using hlsl202x instead}}
// expected-warning@#pi{{variable templates are a HLSL 202x extension}}
// expected-warning@#zero{{variable templates are a HLSL 202x extension}}
// expected-warning@#arr{{variable templates are a HLSL 202x extension}}
// expected-warning@#max{{variable templates are a HLSL 202x extension}}
#endif

template <typename T> static const T Pi = T(3.14159265); // #pi
template <> const int Pi<int> = 3;

template <typename T> static T Zero = T(0); // #zero

template <typename T, uint N> static const T Arr[N] = {T(1), T(2)}; // #arr

struct S {
  template <typename T> static const T Max = T(100); // #max
};

_Static_assert(Pi<int> == 3, "explicit specialization");
_Static_assert(Pi<uint> == 3, "implicit instantiation");
_Static_assert(S::Max<int> == 100, "static data member template");

template <typename T> T twice() { return Pi<T> * 2; }

export float fn(uint I) {
  Zero<float> += 1.0;
  return twice<float>() + Zero<float> + Arr<float, 2>[I];
}

export float bad() {
  // expected-error@+1 {{use of variable template 'Pi' requires template arguments}}
  return Pi;
  // expected-note@#pi {{template is declared here}}
}
