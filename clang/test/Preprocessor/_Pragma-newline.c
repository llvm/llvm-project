// RUN: %clang_cc1 -E -o - %s | FileCheck %s
// RUN: %clang_cc1 -E -P -o - %s | FileCheck %s
// RUN: %clang_cc1 -E -fminimize-whitespace -o - %s | FileCheck %s
// RUN: %clang_cc1 -E -fminimize-whitespace -P -o - %s | FileCheck %s

// The PragmaAssumeNonNullHandler (and maybe others) passes an invalid
// SourceLocation when inside a _Pragma. Ensure we still emit semantic
// newlines.
// See report at https://reviews.llvm.org/D104601#3105044

_Pragma("clang assume_nonnull begin") test _Pragma("clang assume_nonnull end")

// CHECK: {{^}}#pragma clang assume_nonnull begin{{$}}
// CHECK: test
// CHECK: {{^}}#pragma clang assume_nonnull end{{$}}
