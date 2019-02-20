// REQUIRES: x86-registered-target

// Testing to ensure -enable-split-dwarf=single allows to place .dwo sections into regular output object.
//  RUN: %clang_cc1 -debug-info-kind=limited -triple x86_64-unknown-linux \
//  RUN:   -enable-split-dwarf=single -split-dwarf-file %t.o -emit-obj -o %t.o %s
//  RUN: llvm-objdump -section-headers %t.o | FileCheck --check-prefix=MODE-SINGLE %s
//  MODE-SINGLE: .dwo

// Testing to ensure -enable-split-dwarf=split does not place .dwo sections into regular output object.
//  RUN: %clang_cc1 -debug-info-kind=limited -triple x86_64-unknown-linux \
//  RUN:   -enable-split-dwarf=split -split-dwarf-file %t.o -emit-obj -o %t.o %s
//  RUN: llvm-objdump -section-headers %t.o | FileCheck --check-prefix=MODE-SPLIT %s
//  MODE-SPLIT-NOT: .dwo

int main (void) {
  return 0;
}
