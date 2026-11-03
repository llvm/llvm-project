// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

void testAsmGoto() {
  asm goto("xor %0, %0\n je %l[label1]\n jl %l[label2]"
           : /* no outputs */
           : /* inputs */
           : /* clobbers */
           : label1, label2 /* any labels used */);

  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}

  label1:
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return;

  label2:
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return;
}
