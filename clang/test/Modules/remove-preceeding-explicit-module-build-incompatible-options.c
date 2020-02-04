// RUN: rm -rf %t
// RUN: %clang_cc1 -emit-obj -o %t.o input-that-doesnt-exist.c \
// RUN:   -remove-preceeding-explicit-module-build-incompatible-options \
// RUN:   -emit-llvm-bc -o %t.bc %s
// RUN: not ls %t.o
// RUN: llvm-dis %t.bc
