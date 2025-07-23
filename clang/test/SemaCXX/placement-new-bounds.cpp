// RUN: %clang -fsyntax-only -Xclang -analyze -Xclang -analyzer-checker=cplusplus.PlacementNew -Xclang -verify -std=c++17 %s

#include <new>

void test_exact_size() {
    void *buf = ::operator new(sizeof(int)*2);
    int *placement_int = new (buf) int[2]; // no-warning
    placement_int[0] = 42;
    ::operator delete(buf);
}

void test_undersize() {
    void *buf = ::operator new(sizeof(int)*1);
    int *placement_int = new (buf) int[2]; // expected-warning {{Storage provided to placement new is only 4 bytes, whereas the allocated type requires 8 bytes [cplusplus.PlacementNew]}}
    placement_int[0] = 42;
    ::operator delete(buf);
}

void test_oversize() {
    void *buf = ::operator new(sizeof(int)*4);
    int *placement_int = new (buf) int[2]; // no-warning
    placement_int[0] = 42;
    ::operator delete(buf);
}