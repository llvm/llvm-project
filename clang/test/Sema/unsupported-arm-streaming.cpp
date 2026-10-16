// RUN: %clang_cc1 -x c -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

void arm_streaming(void) __arm_streaming {} // expected-error {{'__arm_streaming' is not supported on this target}}
