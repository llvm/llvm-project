// RUN: rm -rf %t && mkdir %t
// RUN: echo 'module Foo { header "foo.h" }' > %t/module.modulemap
// RUN: echo '#define FOO_MACRO 42' > %t/foo.h
// RUN: c-index-test -code-completion-at=%s:%(line+5):1 -I %t %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:%(line+4):1 -I %t -fmodules -fmodules-cache-path=%t %s | FileCheck %s

#include "foo.h"
int x =
/*here*/1;

// CHECK: FOO_MACRO
