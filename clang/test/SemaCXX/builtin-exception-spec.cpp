// RUN: %clang_cc1 -isystem %S/Inputs -fsyntax-only -verify %s
// RUN: %clang_cc1 -isystem %S/Inputs -fsyntax-only -verify -std=c++1z %s
// expected-no-diagnostics
#include <malloc.h>

extern "C" {
void *malloc(__SIZE_TYPE__);
}
