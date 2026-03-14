// RUN: %clang_cc1 -triple thumbv7--windows -fms-extensions -fsyntax-only -verify %s
// expected-no-diagnostics

__declspec(thread) int i;

