int foo(int x) __asm__("_foo_");

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: asm-attribute.c:1:5: FunctionDecl=foo:1:5 Extent=[1:1 - 1:32]
// CHECK: asm-attribute.c:1:24: asm label=_foo_ Extent=[1:24 - 1:31]
