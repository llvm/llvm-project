// RUN: %clang_cc1 -extract-api -triple arm64-apple-macosx \
// RUN:   -x c-header %s -o - -verify | FileCheck %s

// Global record
int _HiddenGlobal;
int exposed_global;

// Record type
struct _HiddenRecord {
  int HiddenRecordMember;
};

struct ExposedRecord {
  int ExposedRecordMember;
};

// Macros
#define _HIDDEN_MACRO 5
#define EXPOSED_MACRO 5

// expected-no-diagnostics

// CHECK-NOT: _HiddenRecord
// CHECK-NOT: HiddenRecordMember
// CHECK: ExposedRecord
// CHECK: ExposedRecordMember
// CHECK-NOT: _HIDDEN_MACRO
// CHECK: EXPOSED_MACRO
