// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fsyntax-only -verify %s

// expected-error@+1 {{'patchable_function_entry' attribute is not yet supported on AIX}}
__attribute__((patchable_function_entry(0))) void f();
