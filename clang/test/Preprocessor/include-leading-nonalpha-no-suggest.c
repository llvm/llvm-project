// RUN: %clang_cc1 %s -verify

#include "/non_existing_file_to_include.h" // expected-error {{'/non_existing_file_to_include.h' file not found}}
