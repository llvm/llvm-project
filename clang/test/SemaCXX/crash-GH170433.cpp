// RUN: %clang_cc1 -verify %s

// https://github.com/llvm/llvm-project/issues/170433

// expected-no-diagnostics

void f(double);

template <class = int>
void f(double);

_Atomic double atomic_value = 42.5;

void test() {
    f(atomic_value);
}
