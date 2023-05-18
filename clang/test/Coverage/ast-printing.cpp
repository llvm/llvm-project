// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++20 -ast-print %s -o %t.1.cpp
// RUN: %clang_cc1 -std=c++20 -ast-print %t.1.cpp -o %t.2.cpp
// RUN: diff %t.1.cpp %t.2.cpp
// RUN: %clang_cc1 -std=c++20 -ast-dump %s
// RUN: %clang_cc1 -std=c++20 -ast-dump-all %s
// RUN: %clang_cc1 -std=c++20 -ast-dump=json -triple=x86_64-linux-gnu %s
// RUN: %clang_cc1 -std=c++20 -fdump-record-layouts %s

#include "cxx-language-features.inc"
