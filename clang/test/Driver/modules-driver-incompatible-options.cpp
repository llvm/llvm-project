// Checks for diagnostics that report incompatibilities between
// -fmodules-driver and other options.

// RUN: split-file %s %t
// RUN: not %clang -std=c++20 -fmodules-driver -fno-modules-reduced-bmi main.cpp -###

// CHECK: clang: error: '-fmodules-driver' is currently incompatible with '-fno-modules-reduced-bmi'

//--- main.cpp
int main() {}
