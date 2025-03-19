// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsyntax-only -verify -Wpadded %s

struct __attribute__((ms_struct)) BitfieldStruct { // expected-warning {{padding size of 'BitfieldStruct' with 3 bytes to alignment boundary}}
  char c : 1;
  int : 0; // expected-warning {{padding struct 'BitfieldStruct' with 31 bits to align anonymous bit-field}}
  char i;
};

int main() {
  BitfieldStruct b;
}
