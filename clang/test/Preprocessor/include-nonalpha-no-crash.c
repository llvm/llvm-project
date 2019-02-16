// RUN: %clang_cc1 %s -verify

#include "./" // expected-error {{'./' file not found}}
