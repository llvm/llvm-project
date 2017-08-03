void function() {
struct Class {
  // one-method: +2:3
  // all-methods-begin: +1:1
  Class();

  ~Class(); // comment

  int constOverride() const override;

  // comment
  void method(const int &value, int defaultParam = 20)
      ;

  void implementedMethod() const {

  }
};
// all-methods-end: -1:1
}
// CHECK1: " { \n  <#code#>;\n}" [[@LINE-16]]:10 -> [[@LINE-16]]:11
// RUN: clang-refactor-test perform -action implement-declared-methods -at=one-method  %s | FileCheck --check-prefix=CHECK1 %s

// CHECK2: " { \n  <#code#>;\n}" [[@LINE-19]]:10 -> [[@LINE-19]]:11
// CHECK2-NEXT: "" [[@LINE-18]]:11 -> [[@LINE-18]]:12
// CHECK2-NEXT: " { \n  <#code#>;\n}" [[@LINE-19]]:23 -> [[@LINE-19]]:23
// CHECK2-NEXT: " { \n  <#code#>;\n}" [[@LINE-18]]:37 -> [[@LINE-18]]:38
// CHECK2-NEXT: " { \n  <#code#>;\n}" [[@LINE-15]]:7 -> [[@LINE-15]]:8
// RUN: clang-refactor-test perform -action implement-declared-methods -selected=all-methods  %s | FileCheck --check-prefix=CHECK2 %s

