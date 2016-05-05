// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:                -I %S/Inputs/submodules %s 2>&1 | FileCheck %s
// CHECK: import of module 'import_self.c' appears within same top-level module 'import_self'

// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:                -I %S/Inputs/submodules -fmodule-name=import_self %s \
// RUN:     2>&1 |  FileCheck -check-prefix=CHECK-fmodule-name %s
// CHECK-fmodule-name: import of module 'import_self.b' appears within same top-level module 'import_self'

@import import_self.b;
