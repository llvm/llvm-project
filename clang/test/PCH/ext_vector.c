// Test this without pch.
// RUN: %clang_cc1 -include %S/ext_vector.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/ext_vector.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

int test(float4 f4) {
  return f4.xy; // expected-error{{float2}}
}
