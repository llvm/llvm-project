// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST1
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST2
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST3
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST4
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST5
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST6
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST7
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST8
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST9 -ffreestanding

#if TEST1
int main; // expected-warning{{variable named 'main' with external linkage has undefined behavior}}

#elif TEST2
extern int main; // expected-warning{{variable named 'main' with external linkage has undefined behavior}}

#elif TEST3
// expected-no-diagnostics
void x(void) {
  static int main;
}

#elif TEST4
void x(void) {
  extern int main; // expected-warning{{variable named 'main' with external linkage has undefined behavior}}
}

#elif TEST5
// expected-no-diagnostics
void x(void) {
  int main;
}

#elif TEST6
// expected-no-diagnostics
static int main;

#elif TEST7
// expected-no-diagnostics
void x(void) {
  auto int main;
}

#elif TEST8
// expected-no-diagnostics
void x(void) {
  register int main;
}

#elif TEST9
// expected-no-diagnostics
int main;

#else
#error Unknown Test
#endif
