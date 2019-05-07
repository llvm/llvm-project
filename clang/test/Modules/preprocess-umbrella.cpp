// FIXME: The standalone module still seems to cause clang to want to test for
// the existence of a 'foo' directory:
// RUN: mkdir %t
// RUN: cp %s %t
// RUN: mkdir %t/foo
// RUN: cd %t
// RUN: not %clang_cc1 -fmodules -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: error: no matching function for call to 'foo'
// CHECK: note: candidate function not viable: requires 0 arguments, but 1 was provided

// FIXME: This should use -verify, but it seems it doesn't hook up the
// SourceManager correctly or something, and the foo.h note gets attributed to
// the synthetic module translation unit "foo.map Line 2:...".
// %clang_cc1 -fmodules -verify %s

#pragma clang module build foo
module foo {
  umbrella "foo"
  module * {
    export *
  }
}
#pragma clang module contents
#pragma clang module begin foo.foo
# 1 "foo.h" 1
#ifndef FOO_FOO_H
void foo();
#endif
#pragma clang module end
#pragma clang module endbuild
#pragma clang module import foo.foo
// expected-note@foo.h:2 {{candidate function not viable: requires 0 arguments, but 1 was provided}}
int main() {
  foo(1); // expected-error {{no matching function for call to 'foo'}}
}
