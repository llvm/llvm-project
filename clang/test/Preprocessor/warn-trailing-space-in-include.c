// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-warning@+2 {{trailing space in #include file name}}
// expected-error@+1 {{'stdio.h' file not found}}
#include <stdio.h >
#include "stdio.h "