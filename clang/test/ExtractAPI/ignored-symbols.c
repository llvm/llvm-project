// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -extract-api -triple arm64-apple-macosx \
// RUN:   --extract-api-ignores=%t/ignores-list            \
// RUN:   -x c-header %t/input.h -verify -o - | FileCheck %t/input.h

//--- input.h
#define IGNORED_1 1
#define IGNORED_2 2
#define IGNORED_3 3
#define IGNORED_4 4
typedef int Ignored;
typedef float NonIgnored;

// CHECK-NOT: IGNORED_1
// CHECK-NOT: IGNORED_2
// CHECK-NOT: IGNORED_3
// CHECK: NonIgnored

// expected-no-diagnostics

//--- ignores-list
Ignored
IGNORED_4
IGNORED_3   
IGNORED_2
IGNORED_1
