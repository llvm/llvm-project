

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
// expected-no-diagnostics

// This file exercises buildAndChainOldCountCheck for nested MemberExprs 

struct nested_fam {
    int dummy;
    int len;
};
struct outer {
    struct nested_fam hdr2;
    int dummy;
};
struct outerouter {
    struct outer hdr;
    char fam[__counted_by(hdr.hdr2.len)];
};

struct outerouter s = {
    .hdr.hdr2.len = 2,
    .fam = {99,98}
};

void foo() {
    s.hdr.hdr2.len = 1;
}

void bar() {
    // rdar://127523062 error here
    struct outerouter t;
    t.hdr.hdr2.len = 8;
}
