// RUN: %clang_cc1 -fsyntax-only %s
// RUN: %clang_cc1 -ast-print %s -o %t.1.cpp
// RUN: %clang_cc1 -ast-print %t.1.cpp -o %t.2.cpp
// RUN: diff %t.1.cpp %t.2.cpp
// RUN: %clang_cc1 -ast-dump %s
// RUN: %clang_cc1 -print-decl-contexts %s
// RUN: %clang_cc1 -fdump-record-layouts %s

#include "cxx-language-features.inc"
