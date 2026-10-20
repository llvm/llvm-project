// RUN: clang-reorder-fields --extra-arg="-std=c++17" -record-name Foo -fields-order z,y,x %s -- 2>&1 | FileCheck --check-prefix=CHECK-MESSAGES %s
// RUN: clang-reorder-fields --extra-arg="-std=c++17" -record-name Foo -fields-order z,y,x %s -- | FileCheck %s

// The order of fields should not change.
class Foo {
public:
  int x;  // CHECK:       {{^  int x;}}
  int y;  // CHECK-NEXT:  {{^  int y;}}
  int z;  // CHECK-NEXT:  {{^  int z;}}
};

int main() {
  // CHECK-MESSAGES: :[[@LINE+1]]:13: Only full initialization without implicit values is supported
  Foo foo = { 0, 1 }; // CHECK: {{^  Foo foo = { 0, 1 };}}
  return 0;
}
