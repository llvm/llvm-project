// RUN: %clang_cc1 -std=c99 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-c99.cir
// RUN: FileCheck --input-file=%t-c99.cir %s --check-prefixes=CIR,C99
// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-c11.cir
// RUN: FileCheck --input-file=%t-c11.cir %s --check-prefixes=CIR,C11
// RUN: %clang_cc1 -std=c99 -ffinite-loops -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-finite.cir
// RUN: FileCheck --input-file=%t-finite.cir %s --check-prefixes=CIR,FINITE
// RUN: %clang_cc1 -std=c11 -fno-finite-loops -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-nofinite.cir
// RUN: FileCheck --input-file=%t-nofinite.cir %s --check-prefixes=CIR,NOFINITE

// In C11 a loop must make progress unless its controlling expression is a
// constant. C99 loops don't, unless -ffinite-loops says so, and
// -fno-finite-loops turns it off. A trivial infinite loop never does.

void use(int);

void f(int n) {
  // CIR-LABEL: cir.func {{.*}} @f
  // CIR: cir.for
  for (int i = 0; i < n; ++i)
    use(i);
  // C99-NOT: loop_hints
  // C11: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // FINITE: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // NOFINITE-NOT: loop_hints

  // CIR: cir.while
  while (n)
    --n;
  // C11: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // FINITE: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>

  // CIR: cir.do
  do {
    --n;
  } while (n > 0);
  // C11: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
  // FINITE: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>

  // With a constant condition, only -ffinite-loops marks the loops, unless they
  // are trivially empty.
  // CIR: cir.while
  while (1)
    use(n);
  // FINITE: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>

  // CIR: cir.for
  for (;;)
    use(n);
  // FINITE: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>

  // CIR: cir.while
  while (1) {
  }
  // A loop without any hints doesn't have a loop id either.
  // CIR-NOT: loop_id
  // CIR-NOT: loop_hints
  // CIR: cir.return
}

