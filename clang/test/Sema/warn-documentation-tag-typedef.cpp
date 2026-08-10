// RUN: %clang_cc1 -Wdocumentation -fsyntax-only %s 2>&1 | FileCheck -allow-empty %s

/*!
@class Foo
*/
typedef class { } Foo;
// CHECK-NOT: warning:

/*! 
@struct Bar
*/
typedef struct { } Bar;
// CHECK-NOT: warning:
