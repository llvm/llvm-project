// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

// expected-no-diagnostics

void clang_analyzer_warnIfReached();

void testAsmGoto() {
  asm goto("xor %0, %0\n je %l[label1]\n jl %l[label2]"
           : /* no outputs */
           : /* inputs */
           : /* clobbers */
           : label1, label2 /* any labels used */);

  // FIXME: Should be reachable.
  clang_analyzer_warnIfReached();

  label1:
  // FIXME: Should be reachable.
  clang_analyzer_warnIfReached();
  return;

  label2:
  // FIXME: Should be reachable.
  clang_analyzer_warnIfReached();
  return;
}
