// RUN: %clang_cc1 -include %S/cmdline-include1.h -x c-header %S/cmdline-include2.h -emit-pch -o %t
// RUN: %clang_cc1 %s -include-pch %t -fsyntax-only -verify
// RUN: %clang_cc1 -x c-header %S/cmdline-include1.h -emit-pch -o %t
// RUN: %clang_cc1 %s -include-pch %t -include %S/cmdline-include2.h -fsyntax-only -verify
// expected-no-diagnostics

int g = x1 + x2;
