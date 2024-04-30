// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library %s -verify
void fn(in out float f); // #fn

// expected-error@#fn2{{duplicate parameter modifier 'in'}}
// expected-note@#fn2{{conflicting attribute is here}}
void fn2(in in float f); // #fn2

// expected-error@#fn3{{duplicate parameter modifier 'out'}}
// expected-note@#fn3{{conflicting attribute is here}}
void fn3(out out float f); // #fn3

// expected-error@#fn4{{duplicate parameter modifier 'in'}}
// expected-error@#fn4{{duplicate parameter modifier 'out'}}
// expected-note@#fn4{{conflicting attribute is here}}
// expected-note@#fn4{{conflicting attribute is here}}
void fn4(inout in out float f); // #fn4

// expected-error@#fn5{{duplicate parameter modifier 'in'}}
// expected-note@#fn5{{conflicting attribute is here}}
void fn5(inout in float f); // #fn5

// expected-error@#fn6{{duplicate parameter modifier 'out'}}
// expected-note@#fn6{{conflicting attribute is here}}
void fn6(inout out float f); // #fn6

// expected-error@#fn-def{{conflicting parameter qualifier 'out' on parameter 'f'}}
// expected-note@#fn{{previously declared as 'inout' here}}
void fn(out float f) { // #fn-def
  f = 2;
}

// Overload resolution failure.
void fn(in float f); // #fn-in

void failOverloadResolution() {
  float f = 1.0;
  fn(f); // expected-error{{call to 'fn' is ambiguous}}
  // expected-note@#fn-def{{candidate function}}
  // expected-note@#fn-in{{candidate function}}
}

void implicitFn(float f);
void inFn(in float f);
void inoutFn(inout float f); // #inoutFn
void outFn(out float f); // #outFn

void callFns() {
  // Call with literal arguments.
  implicitFn(1); // Ok.
  inFn(1); // Ok.
  inoutFn(1); // expected-error{{no matching function for call to 'inoutFn'}}
  // expected-note@#inoutFn{{candidate function not viable: no known conversion from 'int' to 'float &' for 1st argument}}
  outFn(1); // expected-error{{no matching function for call to 'outFn}}
  // expected-note@#outFn{{candidate function not viable: no known conversion from 'int' to 'float &' for 1st argument}}
  
  // Call with variables.
  float f;
  implicitFn(f); // Ok.
  inFn(f); // Ok.
  inoutFn(f); // Ok.
  outFn(f); // Ok.
}

// No errors on these scenarios.

// Alternating `inout` and `in out` spellings between declaration and
// definitions is fine since they have the same semantic meaning.
void fn7(inout float f);
void fn7(in out float f) {}

void fn8(in out float f);
void fn8(inout float f) {}

// These two declare two different functions (although calling them will be
// ambiguous). This is equivalent to declaring a function that takes a
// reference and a function that takes a value of the same type.
void fn9(in float f);
void fn9(out float f);

// The `in` attribute is effectively optional. If no attribute is present it is
// the same as `in`, so these declarations match the functions.
void fn10(in float f);
void fn10(float f) {}

void fn11(float f);
void fn11(in float f) {}

template <typename T>
void fn12(inout T f);

void fn13() {
  float f;
  fn12<float>(f);
}
