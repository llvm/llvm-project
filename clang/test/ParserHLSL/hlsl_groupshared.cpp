// RUN: %clang_cc1 %s -verify
extern groupshared float f; // expected-error{{unknown type name 'groupshared'}}

extern float groupshared f2; // expected-error{{expected ';' after top level declarator}}

namespace {
float groupshared [[]] f3 = 12; // expected-error{{expected ';' after top level declarator}}
}

// expected-error@#fgc{{expected ';' after top level declarator}}
// expected-error@#fgc{{a type specifier is required for all declarations}}
float groupshared const f4 = 12; // #fgc
