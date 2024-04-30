// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify -Wunused %s

__attribute__((target("sse3")))
static int not_used_fmv() { return 1; }
__attribute__((target("avx2")))
static int not_used_fmv() { return 2; }
__attribute__((target("default")))
static int not_used_fmv() { return 0; } // expected-warning {{unused function 'not_used_fmv'}}

__attribute__((target("sse3")))
static int definitely_used_fmv() { return 1; }
__attribute__((target("avx2")))
static int definitely_used_fmv() { return 2; }
__attribute__((target("default")))
static int definitely_used_fmv() { return 0; }
int definite_user() { return definitely_used_fmv(); }
