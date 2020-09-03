// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@interface TypeWithPropertiesBackedByIvars {
  int _bar;
  int _foo;
}
@property(nonatomic) int foo;
@property(nonatomic) int bar;
@end

int getFoo(id object) {
  TypeWithPropertiesBackedByIvars *model = (TypeWithPropertiesBackedByIvars *)object;
  int foo = model.;
  return foo;
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:14:19 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1-NOT: [#int#]_bar
// CHECK-CC1-NOT: [#int#]_foo
// CHECK-CC1: [#int#]bar
// CHECK-CC1: [#int#]foo
