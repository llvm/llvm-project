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

// CHECK: <key>name</key><string>SET_PTR_VAR_TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = 0</string>

#define NULL 0
#define SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO \
  ptr = NULL

void nonFunctionLikeNestedMacroTest() {
  int *ptr;
  SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO;
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>SET_PTR_VAR_TO_NULL_WITH_NESTED_MACRO</string>
// CHECK-NEXT: <key>expansion</key><string>ptr =0</string>

//===----------------------------------------------------------------------===//
// Tests for function-like macro expansions.
//===----------------------------------------------------------------------===//

void setToNull(int **vptr) {
  *vptr = nullptr;
}

#define TO_NULL(x) \
  setToNull(x)

void functionLikeMacroTest() {
  int *ptr;
  TO_NULL(&ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// TODO: Expand arguments.
// CHECK: <key>name</key><string>TO_NULL</string>
// CHECK: <key>expansion</key><string>setToNull(x)</string>

#define DOES_NOTHING(x) \
  {                     \
    int b;              \
    b = 5;              \
  }                     \
  print(x)

#define DEREF(x)   \
  DOES_NOTHING(x); \
  *x

void functionLikeNestedMacroTest() {
  int *a;
  TO_NULL(&a);
  DEREF(a) = 5; // expected-warning{{Dereference of null pointer}}
}

// TODO: Expand arguments.
// CHECK: <key>name</key><string>TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>setToNull(x)</string>

// TODO: Expand arguments.
// CHECK: <key>name</key><string>DEREF</string>
// CHECK-NEXT: <key>expansion</key><string>{ int b; b = 5; } print(x); *x</string>

//===----------------------------------------------------------------------===//
// Tests for undefining and/or redifining macros.
//===----------------------------------------------------------------------===//

#define WILL_UNDEF_SET_NULL_TO_PTR(ptr) \
  ptr = nullptr;

void undefinedMacroByTheEndOfParsingTest() {
  int *ptr;
  WILL_UNDEF_SET_NULL_TO_PTR(ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

#undef WILL_UNDEF_SET_NULL_TO_PTR

// TODO: Expand arguments.
// CHECK: <key>name</key><string>WILL_UNDEF_SET_NULL_TO_PTR</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = nullptr;</string>

#define WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL(ptr) \
  /* Nothing */
#undef WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL
#define WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL(ptr) \
  ptr = nullptr;

void macroRedefinedMultipleTimesTest() {
  int *ptr;
  WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL(ptr)
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

#undef WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL
#define WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL(ptr)                      \
  print("This string shouldn't be in the plist file at all. Or anywhere, " \
        "but here.");

// TODO: Expand arguments.
// CHECK: <key>name</key><string>WILL_REDIFINE_MULTIPLE_TIMES_SET_TO_NULL</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = nullptr;</string>

#define WILL_UNDEF_SET_NULL_TO_PTR_2(ptr) \
  ptr = nullptr;

#define PASS_PTR_TO_MACRO_THAT_WILL_BE_UNDEFD(ptr) \
  WILL_UNDEF_SET_NULL_TO_PTR_2(ptr)

void undefinedMacroInsideAnotherMacroTest() {
  int *ptr;
  PASS_PTR_TO_MACRO_THAT_WILL_BE_UNDEFD(ptr);
  *ptr = 5; // expected-warning{{Dereference of null pointer}}
}

// TODO: Expand arguments.
// CHECK: <key>name</key><string>PASS_PTR_TO_MACRO_THAT_WILL_BE_UNDEFD</string>
// CHECK-NEXT: <key>expansion</key><string>ptr = nullptr;</string>

#undef WILL_UNDEF_SET_NULL_TO_PTR_2
