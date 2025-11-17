// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: ln -s s2.h %t/s2-link.h
// RUN: ln -s s3.h %t/s3-link1.h
// RUN: ln -s s3.h %t/s3-link2.h

// Normal compilation for baseline.
// RUN: %clang_cc1 -x c-header %t/prefix.h -I %t/inc -DCMD_MACRO=1 -emit-pch -o %t/prefix1.pch
// RUN: %clang_cc1 %t/t1.c -include-pch %t/prefix1.pch -emit-llvm -o %t/source.ll -I %t/inc -DCMD_MACRO=1

// RUN: %clang -cc1depscan -o %t/pch.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 -emit-pch -x c-header %t/prefix.h -I %t/inc -DCMD_MACRO=1 -fcas-path %t/cas
// RUN: %clang @%t/pch.rsp -o %t/prefix2.pch

// RUN: %clang -cc1depscan -o %t/tu.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 -emit-llvm %t/t1.c -include-pch %t/prefix2.pch -I %t/inc -DCMD_MACRO=1 -fcas-path %t/cas
// RUN: rm %t/prefix2.pch

// RUN: %clang @%t/tu.rsp -o %t/tree.ll
// RUN: diff -u %t/source.ll %t/tree.ll

// Check again with relative paths.
// RUN: cd %t

// Normal compilation for baseline.
// RUN: %clang_cc1 -x c-header prefix.h -I %t/inc -DCMD_MACRO=1 -emit-pch -o prefix3.pch
// RUN: %clang_cc1 t1.c -include-pch prefix3.pch -emit-llvm -o source-rel.ll -I inc -DCMD_MACRO=1

// RUN: %clang -cc1depscan -o pch2.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 -emit-pch -x c-header prefix.h -I %t/inc -DCMD_MACRO=1 -fcas-path %t/cas
// RUN: %clang @pch2.rsp -o prefix4.pch

// RUN: %clang -cc1depscan -o tu2.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 -emit-llvm t1.c -include-pch prefix4.pch -I inc -DCMD_MACRO=1 -fcas-path %t/cas
// RUN: rm %t/prefix4.pch

// RUN: %clang @tu2.rsp -o tree-rel.ll
// RUN: diff -u source-rel.ll tree-rel.ll

// Check that -coverage-notes-file and -coverage-data-file are stripped
// RUN: %clang -cc1depscan -o pch3.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 -emit-pch -x c-header prefix.h -I %t/inc -DCMD_MACRO=1 -fcas-path %t/cas \
// RUN:     -coverage-notes-file=%t/pch.gcno -coverage-data-file=%t/pch.gcda
// RUN: FileCheck %s -check-prefix=COVERAGE -input-file %t/pch3.rsp
// COVERAGE-NOT: -coverage-data-file
// COVERAGE-NOT: -coverage-notes-file

//--- t1.c
#if S2_MACRO
#include "s2-link.h"
#endif
#include "s3-link2.h"
#include "other.h"

int test(struct S *s, struct S2 *s2) {
  return s->x + s2->y + CMD_MACRO + PREFIX_MACRO + S2_MACRO + S3_MACRO;
}

//--- prefix.h
#include "s2.h"
#include "s3.h"
#include "s3-link1.h"
#include "other.h"

#define PREFIX_MACRO S3_MACRO

struct S {
  int x;
};

//--- s2.h
#pragma once
#define S2_MACRO 3
struct S2 {
  int y;
};

//--- s3.h
#define S3_MACRO 4

//--- inc/other.h
#include "../inc2/other2.h"

//--- inc2/other2.h
