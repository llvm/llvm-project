// RUN: %clang_cc1 -emit-llvm -o %t %s
// PR2743
// <rdr://6094512>

/* CodeGen should handle this even if it makes it past
   sema. Unfortunately this test will become useless once sema starts
   rejecting this. */

static void e0(void);
void f0(void) { e0(); }

inline void e1(void);
void f1(void) { e1(); }

void e2(void) __attribute__((weak));
void f2(void) { e2(); }
