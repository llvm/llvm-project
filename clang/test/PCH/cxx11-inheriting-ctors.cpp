// RUN: %clang_cc1 -std=c++11 -include %s -include %s -verify %s
//
// Emit with definitions in the declaration:
// RxN: %clang_cc1 -std=c++11 -emit-pch -o %t.12 -include %s %s
// RxN: %clang_cc1 -std=c++11 -include-pch %t.12 -verify %s
//
// Emit with definitions in update records:
// RxN: %clang_cc1 -std=c++11 -emit-pch -o %t.1 %s
// RxN: %clang_cc1 -std=c++11 -include-pch %t.1 -emit-pch -o %t.2 -verify %s
// RxN: %clang_cc1 -std=c++11 -include-pch %t.1 -include-pch %t.2 -verify %s


// expected-no-diagnostics

#ifndef HEADER1
#define HEADER1

struct Base {
  Base(int) {}

  template <typename T>
  Base(T) {}
};

struct Test : Base {
  using Base::Base;
};

template <typename T>
struct Test2 : Base {
  using Base::Base;
};

template <typename B>
struct Test3 : B {
  using B::B;
};

#elif !defined(HEADER2)
#define HEADER2

Test test1a(42);
Test test1b(nullptr);
Test2<int> test2a(42);
Test2<int> test2b(nullptr);
Test3<Base> test3a(42);
Test3<Base> test3b(nullptr);

#pragma clang __debug dump Test
#pragma clang __debug dump Test2

#else

Test retest1a(42);
Test retest1b(nullptr);
Test2<int> retest2a(42);
Test2<int> retest2b(nullptr);
Test3<Base> retest3a(42);
Test3<Base> retest3b(nullptr);

#endif
