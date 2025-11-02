// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-pc-linux-gnu

void uses_capture() {
  int x = 42;
  int y = 6;
  auto l = [x, &y]() __attribute__((naked)) { // expected-no-diagnostics
    asm volatile("movl %0, %%eax\n\tmovl %1, %%ebx\n\tretq" : : "r"(x), "r"(y));
  };
  l();
}

void unused_captures() {
  int x = 42;
  auto l = [x]() __attribute__((naked)) { // expected-no-diagnostics
    asm volatile("retq");
  };
  l();
}
