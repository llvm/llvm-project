struct Bar {};
struct Foo {
  void method(Bar bar) {}
};
void NoCrash(Foo t) {
  t.method({.abc = 50}); // CHECK: field designator 'abc' does not refer to any field in type 'Bar'
}
// RUN: c-index-test -index-file %s -Xclang -frecovery-ast 2>&1 | FileCheck %s
