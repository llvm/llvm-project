// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -Wmissing-variable-declarations -fsyntax-only -verify %s
// expected-no-diagnostics

register unsigned long current_stack_pointer asm("rsp");
