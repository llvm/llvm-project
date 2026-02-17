// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -std=hlsl202x -verify -Wconversion %s

groupshared uint SharedData;

void fn1(inout uint Sh) {
  Sh = 5;
}

void fn2(inout groupshared uint Sh);
// expected-error@-1{{'inout' attribute is not compatible with 'groupshared' attribute}}
void fn3(in groupshared uint Sh);
// expected-error@-1{{'in' attribute is not compatible with 'groupshared' attribute}}
void fn4(out groupshared uint Sh);
// expected-error@-1{{'out' attribute is not compatible with 'groupshared' attribute}}
void fn5(groupshared inout uint Sh);
// expected-error@-1{{'inout' attribute is not compatible with 'groupshared' attribute}}
void fn6(groupshared in uint Sh);
// expected-error@-1{{'in' attribute is not compatible with 'groupshared' attribute}}
void fn7(groupshared out uint Sh);
// expected-error@-1{{'out' attribute is not compatible with 'groupshared' attribute}}


template<typename T>
void fn8(inout T A);
// expected-note@-1{{candidate template ignored: substitution failure [with T = groupshared uint]: 'inout' attribute is not compatible with 'groupshared' attribute}}

template void fn8<groupshared uint>(inout groupshared uint A);
// expected-error@-1{{'inout' attribute is not compatible with 'groupshared' attribute}}
// expected-error@-2{{explicit instantiation of 'fn8' does not refer to a function template, variable template, member function, member class, or static data member}}

template<typename T>
void fn9(out T A);
// expected-note@-1{{candidate template ignored: substitution failure [with T = groupshared uint]: 'out' attribute is not compatible with 'groupshared' attribute}}

template void fn9<groupshared uint>(out groupshared uint A);
// expected-error@-1{{'out' attribute is not compatible with 'groupshared' attribute}}
// expected-error@-2{{explicit instantiation of 'fn9' does not refer to a function template, variable template, member function, member class, or static data member}}

template<typename T>
void fn10(in T A);
// expected-note@-1{{candidate template ignored: substitution failure [with T = groupshared uint]: 'in' attribute is not compatible with 'groupshared' attribute}}

template void fn10<groupshared uint>(in groupshared uint A);
// expected-error@-1{{'in' attribute is not compatible with 'groupshared' attribute}}
// expected-error@-2{{explicit instantiation of 'fn10' does not refer to a function template, variable template, member function, member class, or static data member}}

// expected-note@+2{{candidate template ignored: substitution failure [with T = groupshared uint]: 'inout' attribute is not compatible with 'groupshared' attribute}}
template<typename T>
void fn11(inout T A, inout T B) {
  A = B;
}
// expected-note@+2{{candidate template ignored: substitution failure [with T = groupshared uint]: 'out' attribute is not compatible with 'groupshared' attribute}}
template<typename T>
void fn12(out T A, out T B) {
  A = B;
}

// expected-note@+2{{candidate template ignored: substitution failure [with T = groupshared uint]: 'in' attribute is not compatible with 'groupshared' attribute}}
template<typename T>
void fn13(in T A, in T B) {
  A = B;
}

void fn0() {
  fn1(SharedData);
// expected-warning@-1{{passing groupshared variable to a parameter annotated with inout. See 'groupshared' parameter annotation added in 202x}}

  fn11<groupshared uint>(SharedData, SharedData);
// expected-error@-1{{no matching function for call to 'fn11'}}
  fn12<groupshared uint>(SharedData, SharedData);
// expected-error@-1{{no matching function for call to 'fn12'}}
  fn13<groupshared uint>(SharedData, SharedData);
// expected-error@-1{{no matching function for call to 'fn13'}}
}
