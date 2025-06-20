// Check that the -all_load flag to llvm-jitlink causes all objects from
// archives to be loaded, regardless of whether or not they're referenced.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clangxx -c -o %t/SetX.o  %S/Inputs/SetGlobalIntXInConstructor.cpp
// RUN: ar r %t/libSetX.a %t/SetX.o
// RUN: %clang -c -o %t/all_load.o %s
// RUN: %llvm_jitlink -all_load %t/all_load.o -L%t -lSetX
//
// REQUIRES: system-darwin && host-arch-compatible

int x = 0;

int main(int argc, char *argv[]) { return x == 1 ? 0 : 1; }
