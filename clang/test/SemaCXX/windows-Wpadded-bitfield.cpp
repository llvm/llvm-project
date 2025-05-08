// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsyntax-only -verify -Wpadded %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify -Wpadded %s

struct __attribute__((ms_struct)) BitfieldStruct { // expected-warning {{padding size of 'BitfieldStruct' with 3 bytes to alignment boundary}}
  char c : 1;
  int : 0; // expected-warning {{padding struct 'BitfieldStruct' with 31 bits to align anonymous bit-field}}
  char i;
};

struct __attribute__((ms_struct)) SevenBitfieldStruct { // expected-warning {{padding size of 'SevenBitfieldStruct' with 3 bytes to alignment boundary}}
  char c : 7;
  int : 0; // expected-warning {{padding struct 'SevenBitfieldStruct' with 25 bits to align anonymous bit-field}}
  char i;
};

struct __attribute__((ms_struct)) SameUnitSizeBitfield {
  char c : 7;
  char : 1; // Same unit size attributes fall in the same unit + they fill the unit -> no padding
  char i;
};

struct __attribute__((ms_struct)) DifferentUnitSizeBitfield { // expected-warning {{padding size of 'DifferentUnitSizeBitfield' with 3 bytes to alignment boundary}}
  char c : 7;
  int : 1; // expected-warning {{padding struct 'DifferentUnitSizeBitfield' with 25 bits to align anonymous bit-field}}
  char i; // expected-warning {{padding struct 'DifferentUnitSizeBitfield' with 31 bits to align 'i'}}
};

struct __attribute__((ms_struct)) BitfieldBigPadding { // expected-warning {{padding size of 'BitfieldBigPadding' with 63 bits to alignment boundary}}
  long long x;
  char a : 1;
  long long b : 1; // expected-warning {{padding struct 'BitfieldBigPadding' with 63 bits to align 'b'}}
};

struct __attribute__((ms_struct)) SameUnitSizeMultiple { // expected-warning {{padding size of 'SameUnitSizeMultiple' with 2 bits to alignment boundary}}
  char c : 1;
  char cc : 2;
  char ccc : 3;
};

int main() {
  BitfieldStruct b;
  SevenBitfieldStruct s;
  SameUnitSizeBitfield su;
  DifferentUnitSizeBitfield du;
  BitfieldBigPadding bbp;
  SameUnitSizeMultiple susm;
}
