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

static int4 test3() {
  int4 i0, i1;

  return i0 > i1 ? i0 : i1; // no-error
  return i0 ? i0 : i1;      // no-error
}

static float4 test4() {
  float4 f0, f1;

  // This would actually generate implicit casting warning
  // under Weverything flag but we don't really care here
  return f0 > f1 ? f0 : f1; // no-error
  return f0 ? f0 : f1;      // expected-error {{used type 'float4' (vector of 4 'float' values) where floating point type is not allowed}}
}
