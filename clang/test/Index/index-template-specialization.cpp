template <typename T>
class Foo {
public:
  void f(T t) {}
};

void g() {
  Foo<int> foo;
  foo.f(0);
}

// FIXME: if c-index-test uses OrigD for symbol info, refererences below should
// refer to template specialization decls.
// RUN: env CINDEXTEST_INDEXIMPLICITTEMPLATEINSTANTIATIONS=1 c-index-test -index-file %s | FileCheck %s
// CHECK: [indexDeclaration]: kind: c++-class-template | name: Foo
// CHECK-NEXT: [indexDeclaration]: kind: c++-instance-method | name: f
// CHECK-NEXT: [indexDeclaration]: kind: function | name: g
// CHECK-NEXT: [indexEntityReference]: kind: c++-class-template | name: Foo | USR: c:@ST>1#T@Foo
// CHECK-NEXT: [indexEntityReference]: kind: c++-instance-method | name: f | USR: c:@ST>1#T@Foo@F@f#t0.0#
