// RUN: %clang_cc1 %s -verify

// expected-error@#fn{{unknown type name 'in'}}
// expected-error@#fn{{expected ')'}}
// expected-note@#fn{{to match this '('}}
void fn(in out float f); // #fn

// expected-error@#fn2{{unknown type name 'in'}}
// expected-error@#fn2{{expected ')'}}
// expected-note@#fn2{{to match this '('}}
void fn2(in in float f); // #fn2

// expected-error@#fn3{{unknown type name 'out'}}
// expected-error@#fn3{{expected ')'}}
// expected-note@#fn3{{to match this '('}}
void fn3(out out float f); // #fn3

// expected-error@#fn4{{unknown type name 'inout'}}
// expected-error@#fn4{{expected ')'}}
// expected-note@#fn4{{to match this '('}}
void fn4(inout in out float f); // #fn4

// expected-error@#fn5{{unknown type name 'inout'}}
// expected-error@#fn5{{expected ')'}}
// expected-note@#fn5{{to match this '('}}
void fn5(inout in float f); // #fn5

// expected-error@#fn6{{unknown type name 'inout'}}
// expected-error@#fn6{{expected ')'}}
// expected-note@#fn6{{to match this '('}}
void fn6(inout out float f); // #fn6

void implicitFn(float f);

// expected-error@#inFn{{unknown type name 'in'}}
void inFn(in float f); // #inFn

// expected-error@#inoutFn{{unknown type name 'inout'}}
void inoutFn(inout float f); // #inoutFn

// expected-error@#outFn{{unknown type name 'out'}}
void outFn(out float f); // #outFn

// expected-error@#fn7{{unknown type name 'inout'}}
// expected-error@#fn7{{declaration of 'T' shadows template parameter}}
// expected-error@#fn7{{expected ')'}}
// expected-note@#fn7{{to match this '('}}
template <typename T> // expected-note{{template parameter is declared here}}
void fn7(inout T f); // #fn7

