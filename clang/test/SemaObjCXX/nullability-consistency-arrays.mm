// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs %s -D ARRAYS_CHECKED=1 -verify
// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs %s -D ARRAYS_CHECKED=0 -Wno-nullability-completeness-on-arrays -verify
// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs %s -Wno-nullability-completeness -Werror
// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs -x objective-c %s -D ARRAYS_CHECKED=1 -verify
// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs -x objective-c %s -D ARRAYS_CHECKED=0 -Wno-nullability-completeness-on-arrays -verify
// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs -x objective-c %s -Wno-nullability-completeness -Werror

#include "nullability-consistency-arrays.h"

void h1(int *ptr) { } // don't warn

void h2(int * _Nonnull p) { }
