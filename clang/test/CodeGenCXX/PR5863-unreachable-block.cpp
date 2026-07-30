// RUN: %clang_cc1 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -emit-llvm-only %s

// PR5863
class E { };

void P1() {
 try {
  int a=0, b=0;
  if (a > b) // simply filling in 0 or 1 doesn't trigger the assertion
    throw E(); // commenting out 'if' or 'throw' 'fixes' the assertion failure
  try { } catch (...) { } // empty try/catch block needed for failure
 } catch (...) { } // this try/catch block needed for failure
}
