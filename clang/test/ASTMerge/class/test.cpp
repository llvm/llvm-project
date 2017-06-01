// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/class1.cpp
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/class2.cpp
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 -Wno-odr -Werror

// CHECK: class1.cpp:6:8: warning: type 'B' has incompatible definitions in different translation units
// CHECK: class1.cpp:7:9: note: field 'y' has type 'float' here
// CHECK: class2.cpp:7:7: note: field 'y' has type 'int' here

// FIXME: we should also complain about mismatched types on the method

// CHECK: class1.cpp:18:6: warning: type 'E' has incompatible definitions in different translation units
// CHECK: class1.cpp:19:3: note: enumerator 'b' with value 1 here
// CHECK: class2.cpp:12:3: note: enumerator 'a' with value 0 here
