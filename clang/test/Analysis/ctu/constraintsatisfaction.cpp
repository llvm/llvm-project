// RUN: rm -rf %t
// RUN: mkdir -p %t/ctudir
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 \
// RUN:   -emit-pch -o %t/ctudir/import.cpp.ast %t/import.cpp
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -std=c++20 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -verify %t/main.cpp

//--- main.cpp

// expected-no-diagnostics

template <typename T>
concept Sizable = requires(T t) { t.size(); };

template <typename T>
concept Container = Sizable<T> && requires(T t) { t.begin(); };

template <bool> struct BoolConstant {};
using FalseCheck = BoolConstant<Container<int>>;

void importee();
void caller() { importee(); } // no-crash

//--- import.cpp

// Check that importing this code does not cause crash.

template <typename T>
concept Sizable = requires(T t) { t.size(); };

template <typename T>
concept Container = Sizable<T> && requires(T t) { t.begin(); };

template <bool> struct BoolConstant {};
using FalseCheck = BoolConstant<Container<int>>;

void importee() {
  FalseCheck f{};
  (void)f;
}

//--- ctudir/externalDefMap.txt
14:c:@F@importee# import.cpp.ast
