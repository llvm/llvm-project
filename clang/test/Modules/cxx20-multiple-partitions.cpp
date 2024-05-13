// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/partition1.cpp \
// RUN:  -o %t/A_part1.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/partition2.cpp \
// RUN:  -o %t/A_part2.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/partition3.cpp \
// RUN:  -o %t/A_part3.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/moduleA.cpp \
// RUN:  -fmodule-file=%t/A_part1.pcm -fmodule-file=%t/A_part2.pcm \
// RUN:  -fmodule-file=%t/A_part3.pcm -o %t/A.pcm

// expected-no-diagnostics

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

export module A;

import :Part1;
export import :Part2;
import :Part3;

int foo();
