// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -fsyntax-only -verify

namespace TiedAggregateOutput {
class C8 {
  long long a;

public:
  C8();
};

class C12 {
  int a;
  int b;
  int c;

public:
  C12();
};

class C16 {
  long long a;
  long long b;

public:
  C16();
};

void int_tied_to_gpr_sized_class_output() {
  C8 c;
  asm("" : "=rm"(c) : "0"(1)); // no-error
}

void int_tied_to_class_output_too_large() {
  C12 c;
  asm("" : "=rm"(c) : "0"(1)); // expected-error {{impossible constraint in asm: cannot store value into a register}}
}

void int_tied_to_class_output() {
  C16 c;
  asm("" : "=r"(c) : "0"(1)); // expected-error {{impossible constraint in asm: cannot store value into a register}}
}
} // namespace TiedAggregateOutput
