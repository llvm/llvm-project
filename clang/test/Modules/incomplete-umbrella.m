// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F%S/Inputs/incomplete-umbrella -fsyntax-only %s 2>&1 | FileCheck %s

#import <Foo/Foo.h>
#import <Foo/Bar.h>
#import <Foo/Baz.h>
@import Foo.Private;

// CHECK: warning: umbrella header for module 'Foo' does not include header 'Bar.h'
// CHECK: warning: umbrella header for module 'Foo.Private' does not include header 'Baz.h'
int foo() {
  int a = BAR_PUBLIC;
  int b = BAZ_PRIVATE;
  return 0;
}
