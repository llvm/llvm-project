// Test availability message string type when wide characters are 1 byte.
// REQUIRES: xcore-registered-target
// RUN: %clang_cc1 -triple xcore -fsyntax-only -verify %s

#if !__has_feature(attribute_availability)
#  error 'availability' attribute is not available
#endif

void f7() __attribute__((availability(macosx,message=L"wide"))); // expected-error {{expected string literal for optional message in 'availability' attribute}}

void f8() __attribute__((availability(macosx,message="a" L"b"))); // expected-error {{expected string literal for optional message in 'availability' attribute}}
