
// RUN: %clang_cc1 -isystem %S/mock-sdk -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -isystem %S/mock-sdk -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

// expected-no-diagnostics

#include <ptrcheck.h>
#include <array-parameter.h>
