// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu -fsyntax-only -verify -fcf-protection=branch %s
// RUN: %clang_cc1 -triple arm-unknown-linux-gnu -fsyntax-only -verify -fcf-protection=branch %s
// RUN: %clang_cc1 -triple arm-unknown-linux-gnu -fsyntax-only -verify %s

void __attribute__((nocf_check)) foo(void); // expected-warning-re{{{{((unknown attribute 'nocf_check' ignored)|('nocf_check' attribute ignored; use -fcf-protection to enable the attribute))}}}}
