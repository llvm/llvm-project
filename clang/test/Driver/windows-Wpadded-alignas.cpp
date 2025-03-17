// RUN: %clang_cl -Wpadded -Wno-msvc-not-found -fsyntax-only -- %s 2>&1 | FileCheck -check-prefix=WARN %s
// RUN: %clang -Wpadded -fsyntax-only -- %s 2>&1 | FileCheck -check-prefix=WARN %s

struct __attribute__((ms_struct)) AlignedStruct {
  char c;
  alignas(8) int i;
};

int main() {AlignedStruct s;}

// WARN: warning: padding struct 'AlignedStruct' with 7 bytes to align 'i'
// WARN: warning: padding size of 'AlignedStruct' with 4 bytes to alignment boundary
