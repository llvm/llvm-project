// RUN: %clang_cc1 %s -verify
// RUN: %clang_cc1 -fms-compatibility %s -verify

#include "<empty_file_to_include.h>" // expected-error {{'<empty_file_to_include.h>' file not found, did you mean 'empty_file_to_include.h'?}}
