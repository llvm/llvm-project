class Foo; // CHECK: class Bar;
Foo *f();  // CHECK: Bar *f();

// RUN: clang-rename -offset=6 -new-name=Bar %s --  | sed 's,//.*,,' | FileCheck %s
