// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -std=c++20 -x c++ < %s | FileCheck -check-prefix=WIN64 %s

struct A {
  const int* ptr;
};

template<A> void tfn() {};

// WIN64: ??$tfn@$2UA@@PEBH5CE?ints@@3QBHB06@@@@@YAXXZ
constexpr int ints[] = { 1, 2, 7, 8, 9, -17, -10 };

// WIN64: ??$tfn@$2UA@@PEBH5E?one_int@@3HB@@@@YAXXZ
constexpr int one_int = 7;

void template_instance() {
  tfn<A{ints + sizeof(ints)/sizeof(int)}>();
  tfn<A{&one_int + 1}>();
}

