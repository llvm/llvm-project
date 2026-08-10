// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -Wno-explicit-specialization-storage-class -verify=expnone %s

// expnone-no-diagnostics

struct A {
    template<typename T>
    static constexpr int x = 0;

    template<>
    static constexpr int x<void> = 1; // expected-warning{{explicit specialization cannot have a storage class}}
};

template<typename T>
static constexpr int x = 0;

template<>
static constexpr int x<void> = 1; // expected-warning{{explicit specialization cannot have a storage class}}
