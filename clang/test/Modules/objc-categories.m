// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -x objective-c -fmodule-name=category_top -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -x objective-c -fmodule-name=category_left -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -x objective-c -fmodule-name=category_right -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -x objective-c -fmodule-name=category_bottom -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -x objective-c -fmodule-name=category_other -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t %s -verify

@__experimental_modules_import category_bottom;




// in category_left.h: expected-note {{previous definition}}
// in category_right.h: expected-warning@11 {{duplicate definition of category}}

@interface Foo(Source)
-(void)source; 
@end

void test(Foo *foo, LeftFoo *leftFoo) {
  [foo source];
  [foo bottom];
  [foo left];
  [foo right1];
  [foo right2];
  [foo top];
  [foo top2];
  [foo top3];

  [leftFoo left];
  [leftFoo bottom];
}

// Load another module that also adds categories to Foo, verify that
// we see those categories.
@__experimental_modules_import category_other;

void test_other(Foo *foo) {
  [foo other];
}
