// RUN: clang-reorder-fields -record-name Foo -fields-order z,y,x %s -- | FileCheck %s

// The order of fields should not change.
class Foo {
public:
  int x;  // CHECK:       {{^  int x;}}

private:
  int y;  // CHECK:       {{^  int y;}}
  int z;  // CHECK-NEXT:  {{^  int z;}}
};

int main() {
  Foo foo;
  return 0;
}
