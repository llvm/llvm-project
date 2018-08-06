// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unreachable-code %s

typedef __attribute__(( ext_vector_type(4) )) int int4;

static int4 test1() {
  int4 vec, rv;

  // comparisons to self...
  return vec == vec; // expected-warning{{self-comparison always evaluates to true}}
  return vec != vec; // expected-warning{{self-comparison always evaluates to false}}
  return vec < vec; // expected-warning{{self-comparison always evaluates to false}}
  return vec <= vec; // expected-warning{{self-comparison always evaluates to true}}
  return vec > vec; // expected-warning{{self-comparison always evaluates to false}}
  return vec >= vec; // expected-warning{{self-comparison always evaluates to true}}
}


typedef __attribute__(( ext_vector_type(4) )) float float4;

static int4 test2() {
  float4 vec, rv;

  // comparisons to self.  no warning, they're floats
  return vec == vec; // no-warning
  return vec != vec; // no-warning
  return vec < vec;  // no-warning
  return vec <= vec; // no-warning
  return vec > vec;  // no-warning
  return vec >= vec; // no-warning
}
