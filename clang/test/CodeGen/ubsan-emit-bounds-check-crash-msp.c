// REQUIRES: msp430-registered-target
// RUN: %clang -c -fsanitize=undefined -Wno-tentative-definition-array -Wno-return-type -Wno-unused-value -Wno-array-bounds -Xclang -verify --target=msp430-- %s
int a;
_Complex double b[1][1];
void c(void) {
  b[a][8920]; // expected-error {{Expression caused pointer calculation overflow during code generation}}
}
