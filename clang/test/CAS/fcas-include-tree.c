// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 %t/t1.c -E -P -o %t/source.i -isystem %t -DCMD_MACRO=1 -Werror
// RUN: %clang_cc1 %t/t1.c -emit-llvm -o %t/source.ll -isystem %t -DCMD_MACRO=1 -Werror -dependency-file %t/t1-source.d -MT deps

// RUN: %clang -cc1depscan -o %t/inline.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 -E -P %t/t1.c -isystem %t -DCMD_MACRO=1 -fcas-path %t/cas -Werror
// RUN: %clang @%t/inline.rsp -o %t/tree.i
// RUN: diff -u %t/source.i %t/tree.i
// RUN: %clang -cc1depscan -o %t/inline2.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 -emit-llvm %t/t1.c -isystem %t -DCMD_MACRO=1 -fcas-path %t/cas -Werror -dependency-file %t/t1-tree.d -MT deps
// RUN: %clang @%t/inline2.rsp -o %t/tree.ll
// RUN: diff -u %t/source.ll %t/tree.ll
// RUN: diff -u %t/t1-source.d %t/t1-tree.d

//--- t1.c
#include "top.h"
#include "n1.h"
#include <sys.h>

#define N2H <n2.h>
#include N2H

int test(struct S *s) {
  return s->x + gv + gv2 + SOMEVAL + SOMEVAL2 + CMD_MACRO;
}

//--- top.h
#ifndef _TOP_H_
#define _TOP_H_

#if __has_include("n1.h")
#define SOMEVAL 1
#endif

#if __has_include("nonexistent.h")
#define SOMEVAL 7
#else
#define SOMEVAL2 2
#endif

#include "n1.h"

struct S {
  int x;
};

#endif

//--- sys.h
#define SOMECHECK defined(SOMEDEF)
// This triggers warning: macro expansion producing 'defined' has undefined behavior [-Wexpansion-to-defined]
#if SOMECHECK
#endif

//--- n1.h
#ifndef _N1_H_
#define _N1_H_

#pragma once
#pragma clang system_header

int gv;

#endif

//--- n2.h
int gv2;
