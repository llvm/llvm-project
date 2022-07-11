// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -Warray-parameter -verify %s

template <int N>
void func(int i[10]); // expected-note {{previously declared as 'int[10]' here}}

template <int N>
void func(int i[N]); // expected-warning {{argument 'i' of type 'int[N]' with mismatched bound}}

template <int N>
void func(int (&Val)[N]);

template <>
void func<10>(int (&Val)[10]) {
}

static constexpr int Extent = 10;
void funk(int i[10]);
void funk(int i[Extent]); // no-warning
