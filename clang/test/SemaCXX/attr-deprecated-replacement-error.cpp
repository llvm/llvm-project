// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -verify -fsyntax-only -std=c++11 -fms-extensions %s

#if !__has_feature(attribute_deprecated_with_replacement)
#error "Missing __has_feature"
#endif

int a1 [[deprecated("warning", "fixit")]]; // expected-error{{'deprecated' attribute takes no more than 1 argument}}
int a2 [[deprecated("warning", 1)]]; // expected-error{{'deprecated' attribute takes no more than 1 argument}}

int b1 [[gnu::deprecated("warning", "fixit")]]; // expected-error{{'deprecated' attribute takes no more than 1 argument}}
int b2 [[gnu::deprecated("warning", 1)]]; // expected-error{{'deprecated' attribute takes no more than 1 argument}}

__declspec(deprecated("warning", "fixit")) int c1; // expected-error{{'deprecated' attribute takes no more than 1 argument}}
__declspec(deprecated("warning", 1)) int c2; // expected-error{{'deprecated' attribute takes no more than 1 argument}}

int d1 __attribute__((deprecated("warning", "fixit")));
int d2 __attribute__((deprecated("warning", 1))); // expected-error{{'deprecated' attribute requires a string}}
