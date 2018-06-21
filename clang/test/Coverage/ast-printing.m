// RUN: %clang_cc1 -fsyntax-only -fobjc-exceptions %s
// RUN: %clang_cc1 -ast-print -fobjc-exceptions %s
// RUN: %clang_cc1 -ast-dump -fobjc-exceptions %s

#include "objc-language-features.inc"
