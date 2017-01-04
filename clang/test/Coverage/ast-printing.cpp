// RUN: %clang_cc1 -std=c++14 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++14 -ast-print %s
// RUN: %clang_cc1 -std=c++14 -ast-dump %s
// RUN: %clang_cc1 -std=c++14 -print-decl-contexts %s
// RUN: %clang_cc1 -std=c++14 -fdump-record-layouts %s

#include "cxx-language-features.inc"
