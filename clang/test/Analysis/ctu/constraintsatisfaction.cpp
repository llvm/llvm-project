// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 \
// RUN:   -emit-pch -o %t/ctudir/constraintsatisfaction-import.cpp.ast %S/Inputs/constraintsatisfaction-import.cpp
// RUN: cp %S/Inputs/constraintsatisfaction-import.cpp.externalDefMap.ast-dump.txt %t/ctudir/externalDefMap.txt
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -std=c++20 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -verify %s

// Check that importing this code does not cause crash.
// expected-no-diagnostics

template <typename T>
concept Sizable = requires(T t) { t.size(); };

template <typename T>
concept Container = Sizable<T> && requires(T t) { t.begin(); };

template <bool> struct BoolConstant {};
using FalseCheck = BoolConstant<Container<int>>;

void importee();
void caller() { importee(); }
