// RUN: not %clangxx -fsyntax-only -std=c++23 -Xclang -freflection %s 2>&1 | FileCheck %s
//
// CHECK: error: option '-freflection' is only supported when compiling in C++26 mode
