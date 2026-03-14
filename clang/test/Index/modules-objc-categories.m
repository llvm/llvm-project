@import category_top;
@import category_left;

@interface Sub : Foo
- (void)left_sub;
@end

// RUN: rm -rf %t
// RUN: c-index-test -test-load-source local -fmodules -fmodules-cache-path=%t %s -I%S/../Modules/Inputs | FileCheck %s
// CHECK: modules-objc-categories.m:5:9: ObjCInstanceMethodDecl=left_sub:5:9 [Overrides @2:9]
