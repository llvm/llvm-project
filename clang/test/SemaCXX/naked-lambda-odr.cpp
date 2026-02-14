// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -verify %s

void naked_lambda_capture() {
  int x = 42;
  auto l = [x]() __attribute__((naked)) { // expected-error {{naked attribute is incompatible with lambda captures}}
    asm volatile("movl %0, %%eax" : : "r"(x));
    asm volatile("retq");
  };
  l();
}
