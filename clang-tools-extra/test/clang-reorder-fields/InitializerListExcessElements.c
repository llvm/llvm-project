// RUN: clang-reorder-fields -record-name Foo -fields-order y,x %s -- 2>&1 | FileCheck --check-prefix=CHECK-MESSAGES %s
// RUN: clang-reorder-fields -record-name Foo -fields-order y,x %s -- | FileCheck %s

// The order of fields should not change.
struct Foo {
  int x;  // CHECK:      {{^  int x;}}
  int y;  // CHECK-NEXT: {{^  int y;}}
};

int main() {
  // CHECK-MESSAGES: :[[@LINE+1]]:20: Unsupported initializer list
  struct Foo foo = { .y=9, 123, .x=1 }; // CHECK: {{^ struct Foo foo = { .y=9, 123, .x=1 };}}

  return 0;
}
