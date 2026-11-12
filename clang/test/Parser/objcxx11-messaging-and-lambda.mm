// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

#define OBJCLASS(name) // expected-note {{macro 'OBJCLASS' defined here}}

class NSMutableData;

NSMutableData *test() { // expected-note {{to match this '{'}}
  NSMutableData *data = [[[OBJCLASS(NSMutableDataOBJCLASS( alloc] init] autorelease]; // expected-error {{unterminated function-like macro invocation}} \
  // expected-error {{expected ';' at end of declaration}}
  return data;
} // expected-error {{expected expression}} expected-error {{expected '}'}}
