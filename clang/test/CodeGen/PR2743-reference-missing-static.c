// RUN: %clang_cc1 -emit-llvm -o %t %s
// PR2743
// <rdr://6094512>

/* CodeGen should handle this even if it makes it past
   sema. Unfortunately this test will become useless once sema starts
   rejecting this. */

static void e0();
void f0() { e0(); }

inline void e1();
void f1() { e1(); }

void e2() __attribute__((weak));
void f2() { e2(); }
