// RUN: %clang_cc1 -std=c23 %s -E -verify

#embed <> // expected-error {{empty filename}}
#embed "" // expected-error {{empty filename}}
