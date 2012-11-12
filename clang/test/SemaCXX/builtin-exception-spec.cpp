// RUN: %clang_cc1 -isystem %S/Inputs -fsyntax-only -verify %s
// expected-no-diagnostics
#include <malloc.h>

extern "C" {
void *malloc(__SIZE_TYPE__);
}
