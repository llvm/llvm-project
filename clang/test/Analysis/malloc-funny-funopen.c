// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -verify %t/source.c

//--- header.h
#pragma once
#pragma clang system_header

// A non-standard function named "funopen" with four parameters:
void funopen(int *, int *, int *, int *);

// Note that standard funopen() has five parameters:
//FILE *funopen(const void *cookie,
//              int (*readfn)(void *, char *, int),
//              int (*writefn)(void *, const char *, int),
//              off_t (*seekfn)(void *, off_t, int),
//              int (*closefn)(void *));

//--- source.c

#include "header.h"

// expected-no-diagnostics

// This used to trigger an assertion failure through MallocChecker.
void use_funny_funopen(int *a, int *b, int *c, int *d) {
  funopen(a, b, c, d); // no-crash
}
