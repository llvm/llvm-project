// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-dump %s | FileCheck -strict-whitespace %s

namespace MissingSemicolon {
class Foo {
  void f1() = delete
  void g1();
  void f2()
  void g2();
};
// CHECK:      NamespaceDecl {{.*}} MissingSemicolon
// CHECK:      CXXMethodDecl {{.*}} f1 'void ()' delete
// CHECK:      CXXMethodDecl {{.*}} g1 'void ()'
// CHECK:      CXXMethodDecl {{.*}} f2 'void ()'
// CHECK:      CXXMethodDecl {{.*}} g2 'void ()'

} // namespace MissingSemicolon
