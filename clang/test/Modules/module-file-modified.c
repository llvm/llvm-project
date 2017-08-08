// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'int foo = 0;' > %t/a.h
// RUN: echo 'module A { header "a.h" }' > %t/m.modulemap
// RUN: %clang_cc1 -fmodules -emit-module -fmodule-name=A -x c %t/m.modulemap -o %t/m.pcm
// RUN: echo 'int bar;' > %t/a.h
// RUN: not %clang_cc1 -fmodules -fmodule-file=%t/m.pcm -fmodule-map-file=%t/m.modulemap -x c %s -I%t -fsyntax-only 2>&1 | FileCheck %s
#include "a.h"
int foo = 0; // redefinition of 'foo'
// CHECK: fatal error: file {{.*}} has been modified since the module file {{.*}} was built
// REQUIRES: shell
