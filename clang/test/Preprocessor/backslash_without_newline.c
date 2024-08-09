// RUN: %clang_cc1 -x c -E -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c++ -E -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c++ -E -o - %s | FileCheck %s

#include "./backslash_without_newline.h"

// CHECK: A B \ C
A BACKSLASH_WITH_NEWLINE B BACKSLASH_WITHOUT_NEWLINE C
