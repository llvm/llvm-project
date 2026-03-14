// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -I %S/Inputs %s 2>&1 | FileCheck %s
#include "recursive1.h"

// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -emit-module -fmodule-name=recursive1 %S/Inputs/module.modulemap 2>&1 | FileCheck %s

// CHECK:      While building module 'recursive1'{{( imported from .*[/\]recursive.c:3)?}}:
// CHECK-NEXT: While building module 'recursive2' imported from {{.*Inputs[/\]}}recursive1.h:1:
// CHECK-NEXT: In file included from <module-includes>:1:
// CHECK-NEXT: recursive2.h:1:10: fatal error: cyclic dependency in module 'recursive1': recursive1 -> recursive2 -> recursive1
