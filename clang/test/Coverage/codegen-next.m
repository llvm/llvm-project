// RUN: %clang_cc1 -emit-llvm -fobjc-exceptions -triple x86_64-apple-darwin -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -fobjc-exceptions -triple x86_64-apple-darwin -o %t %s

// An error could be seen for targeting x86_64-win32;
//
//     Bitcast requires types of same width
//       %7 = bitcast i32 %6 to i64
//
// Then, This test should be excluded on x86_64-win32.
// REQUIRES: LP64

#include "objc-language-features.inc"
