// RUN: %clang_cc1 -std=c23 -emit-pch -o %t.1.ast %S/Inputs/enum3.c
// RUN: %clang_cc1 -std=c23 -emit-pch -o %t.2.ast %S/Inputs/enum4.c
// RUN: %clang_cc1 -std=c23 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: enum3.c:2:6: warning: type 'enum E1' has incompatible definitions in different translation units
// CHECK: enum4.c:2:6: note: enumeration 'E1' missing fixed underlying type here
// CHECK: enum3.c:2:6: note: enumeration 'E1' has fixed underlying type here
// CHECK: enum3.c:6:6: warning: type 'enum E2' has incompatible definitions in different translation units
// CHECK: enum4.c:6:6: note: enumeration 'E2' has fixed underlying type here
// CHECK: enum3.c:6:6: note: enumeration 'E2' missing fixed underlying type here
// CHECK: enum3.c:11:6: warning: type 'enum E3' has incompatible definitions in different translation units
// CHECK: enum3.c:11:6: note: enumeration 'E3' declared with incompatible fixed underlying types ('long' vs. 'short')
// CHECK: 3 warnings generated

