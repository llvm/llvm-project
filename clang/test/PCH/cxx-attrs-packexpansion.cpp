// Test this without pch.
// RUN: %clang_cc1 -include %s -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -emit-llvm -o - %s

#ifndef HEADER
#define HEADER

template<typename T, typename... Types>
struct static_variant {
    alignas(Types...) T storage[10];
};

#else

struct A {
    static_variant<int> a;
};
struct B {
    static_variant<A> _b;
};

#endif
