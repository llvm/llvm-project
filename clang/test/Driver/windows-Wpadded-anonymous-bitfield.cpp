// RUN: %clang_cl -Wpadded -Wno-msvc-not-found -fsyntax-only -- %s 2>&1 | FileCheck -check-prefix=WARN %s

struct __attribute__((ms_struct)) BitfieldStruct {
  char c : 1;
  int : 0;
  char i;
};

int main() {BitfieldStruct s;}

// WARN: warning: padding struct 'BitfieldStruct' with 31 bits to align anonymous bit-field
// WARN: warning: padding size of 'BitfieldStruct' with 3 bytes to alignment boundary 
