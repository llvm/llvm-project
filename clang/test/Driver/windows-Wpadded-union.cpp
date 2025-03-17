// RUN: %clang_cl -Wpadded -Wno-msvc-not-found -fsyntax-only -- %s 2>&1 | FileCheck -check-prefix=WARN %s
// RUN: %clang -Wpadded -fsyntax-only -- %s 2>&1 | FileCheck -check-prefix=WARN %s

union __attribute__((ms_struct)) Union {
  char c;
  long long u;
};

struct __attribute__((ms_struct)) StructWithUnion {
  char c;
  int : 0;
  Union t;
  short i;
};

int main() { StructWithUnion s; }

// WARN: warning: padding struct 'StructWithUnion' with 7 bytes to align 't'
// WARN: warning: padding size of 'StructWithUnion' with 6 bytes to alignment boundary
