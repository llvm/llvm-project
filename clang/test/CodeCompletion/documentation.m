// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@interface Base
@end

@interface Test : Base
/// Instance!
@property id instanceProp;
/// Class!
@property (class) id classProp;
@end

void test(Test *obj) {
  [obj instanceProp];
  [Test classProp];
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-brief-comments -code-completion-at=%s:15:8 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: instanceProp : [#id#]instanceProp : Instance!
// CHECK-CC1: setInstanceProp: : [#void#]setInstanceProp:<#(id)#> : Instance!

// RUN: %clang_cc1 -fsyntax-only -code-completion-brief-comments -code-completion-at=%s:16:9 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: classProp : [#id#]classProp : Class!
// CHECK-CC2: setClassProp: : [#void#]setClassProp:<#(id)#> : Class!
