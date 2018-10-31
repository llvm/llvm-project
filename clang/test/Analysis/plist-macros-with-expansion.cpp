// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core %s  \
// RUN:   -analyzer-output=plist -o %t.plist \
// RUN:   -analyzer-config expand-macros=true
//
// Check the actual plist output.
//   RUN: cat %t.plist | %diff_plist \
//   RUN:   %S/Inputs/expected-plists/plist-macros-with-expansion.cpp.plist
//
// Check the macro expansions from the plist output here, to make the test more
// understandable.
//   RUN: FileCheck --input-file=%t.plist %s

void print(const void*);

//===----------------------------------------------------------------------===//
// Tests for non-function-like macro expansions.
//===----------------------------------------------------------------------===//

#define SET_PTR_VAR_TO_NULL \
  ptr = 0

void nonFunctionLikeMacroTest() {
  int *ptr;
  SET_PTR_VAR_TO_NULL;
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string></string>
// CHECK-NEXT: <key>expansion</key><string></string>

#define NULL 0
#define SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO \
  ptr = NULL

void nonFunctionLikeNestedMacroTest() {
  int *ptr;
  SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO;
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string></string>
// CHECK-NEXT: <key>expansion</key><string></string>
