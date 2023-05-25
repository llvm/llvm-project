// RUN: %clang_asan_abi  -O2 -c -fsanitize-stable-abi -fsanitize=address -O0 %s -o %t.o
// RUN: %clangxx -c %p/../../../lib/asan_abi/asan_abi.cpp -o asan_abi.o
// RUN: %clangxx -o %t %t.o %libasan_abi asan_abi.o && %run %t 2>&1

int main() { return 0; }
