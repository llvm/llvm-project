// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/partition1.cpp \
// RUN:  -o %t/A-Part1.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/partition2.cpp \
// RUN:  -o %t/A-Part2.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/partition3.cpp \
// RUN:  -o %t/A-Part3.pcm

// RUN: %clang_cc1 -std=c++20 %t/moduleA.cpp -fsyntax-only -verify \
// RUN:  -fprebuilt-module-path=%t

// Test again with reduced BMI
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/partition1.cpp \
// RUN:  -o %t/A-Part1.pcm

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/partition2.cpp \
// RUN:  -o %t/A-Part2.pcm

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/partition3.cpp \
// RUN:  -o %t/A-Part3.pcm

// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %t/moduleA.cpp -fprebuilt-module-path=%t 

//--- partition1.cpp
export module A:Part1;

int part1();

//--- partition2.cpp

export module A:Part2;

int part2();

//--- partition3.cpp

export module A:Part3;

int part3();

//--- moduleA.cpp
// expected-no-diagnostics
export module A;

import :Part1;
export import :Part2;
import :Part3;

int foo();
