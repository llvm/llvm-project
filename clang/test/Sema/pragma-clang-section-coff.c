// RUN: %clang_cc1 -fsyntax-only -verify %s -triple arm64-windows-msvc
// expected-no-diagnostics
#pragma clang section bss = "mybss.1" data = "mydata.1" rodata = "myrodata.1" text = "mytext.1"
#pragma clang section bss="" data="" rodata="" text=""
#pragma clang section

int a;