// RUN: %clang_cc1 -triple powerpc64-ibm-aix -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64 -target-feature +fmv -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv32 -fsyntax-only -verify %s
__attribute__((target("default")))
int f(void) { return 0; }

#ifdef __riscv
__attribute__((target("cpu=sifive-e20")))
int f(void) { return 1; } // expected-error {{'target' attribute cannot be used for multiversioning on current target}}

#elif defined __aarch64__
__attribute__((target("cpu=apple-a18")))
int f(void) { return 1; } // expected-error {{'target' attribute cannot be used for multiversioning on current target}}

#elif defined _AIX
__attribute__((target("cpu=power10")))
int f(void) { return 1; } // expected-error {{'target' attribute cannot be used for multiversioning on current target}}
#endif

int g(void) { return f(); }
