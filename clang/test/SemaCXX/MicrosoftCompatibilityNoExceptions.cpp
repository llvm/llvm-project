// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-compatibility
// expected-no-diagnostics

// PR13153
namespace std {}
class type_info {};
void f() {
  (void)typeid(int);
}
