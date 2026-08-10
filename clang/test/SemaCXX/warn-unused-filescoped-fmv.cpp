// RUN: %clang_cc1 -triple arm64-apple-darwin -fsyntax-only -verify -Wunused -std=c++98 %s
// RUN: %clang_cc1 -triple arm64-apple-darwin -fsyntax-only -verify -Wunused -std=c++14 %s

__attribute__((target_version("fp16")))
static int not_used_fmv(void) { return 1; }
__attribute__((target_version("fp16fml")))
static int not_used_fmv(void) { return 2; }
__attribute__((target_version("default")))
static int not_used_fmv(void) { return 0; } // expected-warning {{unused function 'not_used_fmv'}}


__attribute__((target_version("fp16")))
static int definitely_used_fmv(void) { return 1; }
__attribute__((target_version("fp16fml")))
static int definitely_used_fmv(void) { return 2; }
__attribute__((target_version("default")))
static int definitely_used_fmv(void) { return 0; }
int definite_user(void) { return definitely_used_fmv(); }
