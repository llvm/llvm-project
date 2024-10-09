// REQUIRES: case-insensitive-filesystem
// RUN: %clang_cc1 -Wall %s

#include "empty_file_to_include.h " // expected-warning {{non-portable path}}
#include "empty_file_to_include.h." // expected-warning {{non-portable path}}
#include "empty_file_to_include.h       " // expected-warning {{non-portable path}}
#include "empty_file_to_include.h......." // expected-warning {{non-portable path}}
#include "empty_file_to_include.h . . . " // expected-warning {{non-portable path}}
#include "empty_file_to_include.h.. .. " // expected-warning {{non-portable path}}

#include "empty_file_to_include.h" // No warning