// RUN: %clang_cc1 -fsyntax-only %s
// RUN: %clang_cc1 -ast-print %s
// RUN: %clang_cc1 -ast-dump %s
// RUN: %clang_cc1 -print-decl-contexts %s
// RUN: %clang_cc1 -fdump-record-layouts %s

#include "cxx-language-features.inc"
