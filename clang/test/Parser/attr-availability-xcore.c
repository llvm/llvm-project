// Test availability message string type when wide characters are 1 byte.
// REQUIRES: xcore-registered-target
// RUN: %clang_cc1 -triple xcore -fsyntax-only -verify %s

#if !__has_feature(attribute_availability)
#  error 'availability' attribute is not available
#endif

void f7() __attribute__((availability(macosx,message=L"wide"))); // expected-warning {{encoding prefix 'L' on an unevaluated string literal has no effect}}

void f8() __attribute__((availability(macosx,message="a" L"b"))); // expected-warning {{encoding prefix 'L' on an unevaluated string literal has no effect}}
