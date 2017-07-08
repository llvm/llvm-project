// RUN: clang-import-test -dump-ast -import %S/Inputs/Hierarchy.cpp -expression %s | FileCheck %s

// CHECK: Overrides:{{.*}}Base::foo

void foo() {
  Derived d;
}
