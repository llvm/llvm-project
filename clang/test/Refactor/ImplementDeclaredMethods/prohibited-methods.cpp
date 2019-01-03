// RUN: clang-refactor-test perform -action implement-declared-methods -selected=all-methods %s -std=c++11 | FileCheck %s
// default, deleted and pure methods should not be implemented!

void function() {
struct Class {
// all-methods-begin: +1:1
  Class();
// CHECK: " { \n  <#code#>;\n}" [[@LINE-1]]:10 -> [[@LINE-1]]:11

  Class(const Class &Other) = default;
  Class(const Class &&Other) = delete;

  virtual void pureMethod(int x) = 0;

  virtual void method() const;
// CHECK-NEXT: " { \n  <#code#>;\n}" [[@LINE-1]]:30 -> [[@LINE-1]]:31
};
// all-methods-end: -1:1
}
