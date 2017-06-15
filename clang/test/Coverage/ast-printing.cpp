// RUN: %clang_cc1 -std=c++14 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++14 -ast-print %s -o %t.1.cpp
// RUN: %clang_cc1 -std=c++14 -ast-print %t.1.cpp -o %t.2.cpp
// RUN: diff %t.1.cpp %t.2.cpp
// RUN: %clang_cc1 -std=c++14 -ast-dump %s
// RUN: %clang_cc1 -std=c++14 -ast-dump-all %s
// RUN: %clang_cc1 -std=c++14 -print-decl-contexts %s
// RUN: %clang_cc1 -std=c++14 -fdump-record-layouts %s

#include "cxx-language-features.inc"
