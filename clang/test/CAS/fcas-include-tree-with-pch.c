// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: ln -s s2.h %t/s2-link.h
// RUN: ln -s s3.h %t/s3-link1.h
// RUN: ln -s s3.h %t/s3-link2.h

// Normal compilation for baseline.
// RUN: %clang_cc1 -x c-header %t/prefix.h -DCMD_MACRO=1 -emit-pch -o %t/prefix1.pch
// RUN: %clang_cc1 %t/t1.c -include-pch %t/prefix1.pch -emit-llvm -o %t/source.ll -DCMD_MACRO=1

// RUN: %clang -cc1depscan -o %t/pch.rsp -fdepscan=inline -fdepscan-include-tree -cc1-args \
// RUN:     -cc1 -x c-header %t/prefix.h -DCMD_MACRO=1 -fcas-path %t/cas
// RUN: %clang @%t/pch.rsp -emit-pch -o %t/prefix2.pch

// RUN: %clang -cc1depscan -o %t/tu.rsp -fdepscan=inline -fdepscan-include-tree -cc1-args \
// RUN:     -cc1 %t/t1.c -include-pch %t/prefix2.pch -DCMD_MACRO=1 -fcas-path %t/cas
// RUN: rm %t/prefix2.pch

// RUN: %clang @%t/tu.rsp -emit-llvm -o %t/tree.ll
// RUN: diff -u %t/source.ll %t/tree.ll

//--- t1.c
#if S2_MACRO
#include "s2-link.h"
#endif
#include "s3-link2.h"

int test(struct S *s, struct S2 *s2) {
  return s->x + s2->y + CMD_MACRO + PREFIX_MACRO + S2_MACRO + S3_MACRO;
}

//--- prefix.h
#include "s2.h"
#include "s3.h"
#include "s3-link1.h"

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
