// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/il.cpp
// RUN: %clang_cc1 -ast-merge %t.1.ast -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
// CHECK-NOT: warning: field '' declared with incompatible types in different translation units ('bool' vs. 'int')
