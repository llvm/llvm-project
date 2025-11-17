// RUN: rm -rf %t
// RUN: split-file %s %t/src
// RUN: mkdir %t/out

// RUN: %clang_cc1 -triple x86_64-apple-macos12 -fsyntax-only %t/src/main.c -I %t/src/inc -Wunknown-pragmas 2> %t/regular-diags1.txt

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/t1.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos12 -fcas-path %t/cas -fdepscan-prefix-map %t/src /^src \
// RUN:     -emit-obj %t/src/main.c -o %t/out/output.o -I %t/src/inc -Wunknown-pragmas

// Compare diagnostics after a miss.
// RUN: %clang @%t/t1.rsp 2> %t/diags1.txt
// RUN: diff -u %t/regular-diags1.txt %t/diags1.txt

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/t1.noprefix.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos12 -fcas-path %t/cas \
// RUN:     -emit-obj %t/src/main.c -o %t/out/output.o -I %t/src/inc -Wunknown-pragmas

// Compare diagnostics without prefix mappings.
// RUN: %clang @%t/t1.noprefix.rsp 2> %t/diags1.noprefix.txt
// RUN: diff -u %t/regular-diags1.txt %t/diags1.noprefix.txt

// Check that we have both the remark and source diagnostics.
// RUN: %clang @%t/t1.rsp -Rcompile-job-cache 2> %t/diags-hit1.txt
// RUN: FileCheck %s -input-file %t/diags-hit1.txt
// CHECK: remark: compile job cache hit for
// CHECK: warning: some warning
// CHECK: warning: unknown pragma ignored
// CHECK: warning: using the result of an assignment as a condition without parentheses

// RUN: cat %t/diags-hit1.txt | grep remark: | sed \
// RUN:   -e "s/^.*hit for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key1

// Compare diagnostics after a hit.
// RUN: %clang @%t/t1.rsp 2> %t/cached-diags1.txt
// RUN: diff -u %t/regular-diags1.txt %t/cached-diags1.txt

// RUN: split-file %s %t/src2
// RUN: mkdir %t/out2

// RUN: %clang_cc1 -triple x86_64-apple-macos12 -fsyntax-only %t/src2/main.c -I %t/src2/inc -Wunknown-pragmas 2> %t/regular-diags2.txt

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/t2.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos12 -fcas-path %t/cas -fdepscan-prefix-map %t/src2 /^src \
// RUN:     -emit-obj %t/src2/main.c -o %t/out2/output.o -I %t/src2/inc -Wunknown-pragmas
// RUN: %clang @%t/t2.rsp -Rcompile-job-cache 2> %t/diags-hit2.txt

// RUN: cat %t/diags-hit2.txt | grep remark: | sed \
// RUN:   -e "s/^.*hit for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key2
// RUN: diff -u %t/cache-key1 %t/cache-key2

// RUN: %clang @%t/t2.rsp 2> %t/cached-diags2.txt
// RUN: diff -u %t/regular-diags2.txt %t/cached-diags2.txt

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/terr.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos12 -fcas-path %t/cas -fdepscan-prefix-map %t/src /^src \
// RUN:     -emit-obj %t/src/main.c -o %t/out/output.o -I %t/src/inc -Rcompile-job-cache -DERROR

// RUN: not %clang @%t/terr.rsp -ferror-limit 1 2> %t/diags_error1.txt
// RUN: FileCheck %s -check-prefix=ERROR1 -input-file %t/diags_error1.txt

// ERROR1: error: E1
// ERROR1-NOT: error:
// ERROR1: fatal error: too many errors emitted

// RUN: not %clang @%t/terr.rsp -ferror-limit 2 2> %t/diags_error2.txt
// RUN: FileCheck %s -check-prefix=ERROR2 -input-file %t/diags_error2.txt

// ERROR2: error: E1
// ERROR2: error: E2
// ERROR2-NOT: error:

//--- main.c

#include "t1.h"

#ifdef ERROR
#error E1
#error E2
#endif

//--- inc/t1.h

#warning some warning

_Pragma("unknown1")

#define PRAG(x) _Pragma(x)
PRAG("unknown2")

void test(int x) {
  if (x=0) {}
}

#define DEPR _Pragma("GCC warning \"name is deprecated\"")
#define PARAM "A" DEPR
#define MAC(x) x

void test2() {
  (void)MAC(PARAM);
}
