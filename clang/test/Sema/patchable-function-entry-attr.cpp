// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple aarch64_be -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple i386 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple x86_64 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple loongarch32 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple loongarch64 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple riscv32 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple riscv64 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple ppc64le -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fsyntax-only -verify=AIX %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fsyntax-only -verify=AIX %s

// silence-no-diagnostics

// AIX-error@+2 {{'patchable_function_entry' attribute is not yet supported on AIX}}
// expected-warning@+1 {{unknown attribute 'patchable_function_entry' ignored}}
[[gnu::patchable_function_entry(0)]] void f();
