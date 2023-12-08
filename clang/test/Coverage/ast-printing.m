// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -fsyntax-only -fobjc-exceptions %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -ast-print -fobjc-exceptions %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -ast-dump -fobjc-exceptions %s

#include "objc-language-features.inc"
