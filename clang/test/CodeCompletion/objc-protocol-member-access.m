// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@protocol Bar
@property (readonly) int bar;
@end

@protocol Foo <Bar>

@property (nonatomic, readonly) int foo;
- (void)foobar: (int)x;

@end

int getFoo(id object) {
  id<Foo> modelObject = (id<Foo>)object;
  int foo = modelObject.;
  return foo;
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:17:25 %s -o - | FileCheck %s
// CHECK: bar : [#int#]bar
// CHECK: foo : [#int#]foo
// CHECK-NOT: foobar
