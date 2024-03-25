// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -extract-api -triple arm64-apple-macosx \
// RUN:   --extract-api-ignores=%t/ignores-list1,%t/ignores-list2,%t/ignores-list3 \
// RUN:   -x c-header %t/input.h -verify -o - | FileCheck %t/input.h

//--- input.h
#define IGNORED_6_FILE1 6
#define IGNORED_2_FILE1 2
#define IGNORED_5_FILE1 5

#define IGNORED_4_FILE2 4
#define IGNORED_3_FILE2 3

typedef double IGNORED_1_FILE3;
typedef int IGNORED_7_FILE3;

typedef float NonIgnored;

// CHECK-NOT: IGNORED_6_FILE1
// CHECK-NOT: IGNORED_2_FILE1
// CHECK-NOT: IGNORED_5_FILE1

// CHECK-NOT: IGNORED_4_FILE2
// CHECK-NOT: IGNORED_3_FILE2

// CHECK-NOT: IGNORED_1_FILE3
// CHECK-NOT: IGNORED_7_FILE3
// CHECK: NonIgnored

// expected-no-diagnostics

//--- ignores-list1
IGNORED_6_FILE1
IGNORED_2_FILE1
IGNORED_5_FILE1

//--- ignores-list2
IGNORED_4_FILE2
IGNORED_3_FILE2

//--- ignores-list3
IGNORED_1_FILE3
IGNORED_7_FILE3
