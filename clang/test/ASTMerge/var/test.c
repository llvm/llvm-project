// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/var1.c
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/var2.c
// RUN: not %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only -fdiagnostics-show-note-include-stack %s 2>&1 | FileCheck %s

// CHECK: var2.c:2:9: error: external variable 'x1' declared with incompatible types in different translation units ('double *' vs. 'float **')
// CHECK: var1.c:2:9: note: declared here with type 'float **'
// CHECK: var2.c:3:5: error: external variable 'x2' declared with incompatible types in different translation units ('int' vs. 'double')
// CHECK: In file included from{{.*}}var1.c:3:
// CHECK: var1.h:1:8: note: declared here with type 'double'
// CHECK: error: external variable 'xarray3' declared with incompatible types in different translation units ('int [17]' vs. 'int [18]')
// CHECK: var1.c:7:5: note: declared here with type 'int [18]'
// CHECK: 3 errors
