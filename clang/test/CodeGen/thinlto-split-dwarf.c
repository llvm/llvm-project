// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -debug-info-kind=limited -triple x86_64-unknown-linux-gnu \
// RUN:   -flto=thin -emit-llvm-bc \
// RUN:   -o %t.o %s

// RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
// RUN:   -o %t2.index \
// RUN:   -r=%t.o,main,px

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc \
// RUN:   -o %t.native.o -split-dwarf-file %t.native.dwo -x ir %t.o

// RUN: llvm-readobj -S %t.native.o | FileCheck --check-prefix=O %s
// RUN: llvm-readobj -S %t.native.dwo | FileCheck --check-prefix=DWO %s

// O-NOT: .dwo
// DWO: .dwo

int main() {}
