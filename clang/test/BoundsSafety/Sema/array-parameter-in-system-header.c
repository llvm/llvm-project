
// RUN: %clang_cc1 -isystem %S/mock-sdk -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -isystem %S/mock-sdk -fsyntax-only -fbounds-safety -x objective-c -fbounds-attributes-objc-experimental -verify %s

// expected-no-diagnostics

#include <ptrcheck.h>
#include <array-parameter.h>
