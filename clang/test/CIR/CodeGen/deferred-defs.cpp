// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR --implicit-check-not=unusedExternal \
// RUN:   --implicit-check-not=unusedInternal

extern int usedExternal;
extern int unusedExternal;

int locallyDefined;

namespace {
  int unsedInternal;
  int usedInternal;
}

void use() {
  usedInternal;
  usedExternal;
}

// CIR: cir.global external @locallyDefined = #cir.int<0> : !s32i
// CIR: cir.global "private" internal dso_local @_ZN12_GLOBAL__N_112usedInternalE = #cir.int<0> : !s32i
// CIR: cir.global "private" external @usedExternal : !s32i
