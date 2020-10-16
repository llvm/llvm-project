// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux %s

#define EVAL_EXPR(testno, expr) int test##testno = sizeof(struct{char qq[expr];});

// <rdar://problem/10962435>
EVAL_EXPR(1, ((char*)-1LL) + 1 == 0 ? 1 : -1) // expected-warning {{folded}}
EVAL_EXPR(2, ((char*)-1LL) + 1 < (char*) -1 ? 1 : -1) // expected-warning {{folded}}
