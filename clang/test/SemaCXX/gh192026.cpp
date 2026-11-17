// RUN: %clang_cc1 -fsyntax-only -verify %s

struct ControlSwitcher { bool b; };

class ComplexChain {
  volatile union {
    char flag_byte;
    int ref_count;
  } state_flags[5]; // expected-note {{copy constructor of 'ComplexChain' is implicitly deleted because field 'state_flags' has no copy constructor}}

  ControlSwitcher cs{true};

  ComplexChain trigger_bug() {
    return *this; // expected-error {{call to implicitly-deleted copy constructor of 'ComplexChain'}}
  }
};
