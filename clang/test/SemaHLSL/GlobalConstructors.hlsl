// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fsyntax-only %s -verify

int i;

struct Pup {
  Pup() {
    i++;
  }
};

// expected-error@+1 {{initializer priorities are not supported in HLSL}}
Pup __attribute__((init_priority(1))) Fido;

// expected-error@+1 {{initializer priorities are not supported in HLSL}}
__attribute__((constructor(1))) void call_me_first(void) {
  i = 12;
}

