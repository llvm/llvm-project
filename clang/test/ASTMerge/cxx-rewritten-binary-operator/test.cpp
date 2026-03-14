// RUN: %clang_cc1 -std=c++20 -emit-pch -o %t.1.ast %S/Inputs/rbo.cpp
// RUN: %clang_cc1 -std=c++20 -ast-merge %t.1.ast -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
// CHECK-NOT: unsupported AST node
