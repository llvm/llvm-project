// Make sure we don't trigger use-after-free when we encounter a code completion
// token inside a objc method.
@interface Foo
@end

@implementation Foo
- (void)foo {

// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -code-completion-at=%s:%(line-1):1 %s | FileCheck %s
// CHECK: COMPLETION: self : [#Foo *#]self
  [self foo];
}
@end
