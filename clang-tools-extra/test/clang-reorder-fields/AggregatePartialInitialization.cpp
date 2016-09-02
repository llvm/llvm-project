// RUN: clang-reorder-fields -record-name Foo -fields-order z,y,x %s -- | FileCheck %s

// The order of fields should not change.
class Foo {
public:
  int x;  // CHECK:       {{^  int x;}}
  int y;  // CHECK-NEXT:  {{^  int y;}}
  int z;  // CHECK-NEXT:  {{^  int z;}}
};

int main() {
  Foo foo = { 0, 1 }; // CHECK: {{^  Foo foo = { 0, 1 };}}
  return 0;
}
