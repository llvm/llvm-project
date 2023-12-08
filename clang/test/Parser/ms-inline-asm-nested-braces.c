// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -verify -fasm-blocks

int t_fail(void) { // expected-note {{to match this}}
  __asm
  { // expected-note {{to match this}}
    { // expected-note {{to match this}}
      {
      } // expected-error 3 {{expected}}
