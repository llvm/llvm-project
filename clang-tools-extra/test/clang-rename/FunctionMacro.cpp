#define moo foo           // CHECK: #define moo macro_function

int foo() /* Test 1 */ {  // CHECK: int macro_function() /* Test 1 */ {
  return 42;
}

void boo(int value) {}

void qoo() {
  foo();                  // CHECK: macro_function();
  boo(foo());             // CHECK: boo(macro_function());
  moo();
  boo(moo());
}

// Test 1.
// RUN: clang-rename -offset=68 -new-name=macro_function %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'foo.*' <file>
