// RUN: %clang_cc1 -emit-llvm -fobjc-exceptions -o %t %s
// RUN: %clang_cc1 -g -emit-llvm -fobjc-exceptions -o %t %s

#include "objc-language-features.inc"
