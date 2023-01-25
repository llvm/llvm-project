// RUN: rm -rf %t && mkdir %t
// RUN: echo 'module Foo { header "foo.h" }' > %t/module.modulemap
// RUN: echo '#define FOO_MACRO 42' > %t/foo.h
// RUN: c-index-test -code-completion-at=%s:%(line+4):1 -I %t -fmodules-cache-path=%t -fmodules %s | FileCheck %s

@import Foo;
int x =
/*here*/1;

// CHECK: FOO_MACRO
