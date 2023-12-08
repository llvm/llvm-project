// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F%S/Inputs/incomplete-umbrella -fsyntax-only %s 2>&1 | FileCheck %s

#import <Foo/Foo.h>
#import <Foo/Bar.h>
#import <Foo/Baz.h>
@import Foo.Private;

// CHECK: While building module 'Foo' imported from {{.*[/\]}}incomplete-umbrella.m:4:
// CHECK-NEXT: In file included from <module-includes>:1:
// CHECK-NEXT: {{.*Foo[.]framework[/\]Headers[/\]}}FooPublic.h:2:1: warning: umbrella header for module 'Foo' does not include header 'Bar.h'
// CHECK: While building module 'Foo' imported from {{.*[/\]}}incomplete-umbrella.m:4:
// CHECK-NEXT: In file included from <module-includes>:2:
// CHECK-NEXT: {{.*Foo[.]framework[/\]PrivateHeaders[/\]}}Foo.h:2:1: warning: umbrella header for module 'Foo.Private' does not include header 'Baz.h'
int foo() {
  int a = BAR_PUBLIC;
  int b = BAZ_PRIVATE;
  return 0;
}
