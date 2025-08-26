// RUN: %clang_cc1 -fsyntax-only -verify -x cpp-output %s
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wreserved-macro-identifier"
# 1 "<built-in>" 1
#define __BUILTIN__
# 2 "<command line>" 1
#define __CMD__
# 3 "biz.cpp" 1
#define __SOME_FILE__ // expected-warning {{macro name is a reserved identifier}}
int v;
#pragma clang diagnostic pop
