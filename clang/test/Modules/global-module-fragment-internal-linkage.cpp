// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -emit-module-interface %t/part1.cppm -o %t/A-Part1.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -emit-module-interface %t/part2.cppm -o %t/A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/A-Part2.pcm \
// RUN:   -o %t/A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   %t/use.cpp -fmodule-file=%t/A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/A-Part2.pcm \
// RUN:   | FileCheck %s
//
// Identical internal functions from the same textual header must remain
// distinct when separate global module fragments are imported together.
// CHECK-DAG: define internal {{.*}} @_ZW1AWP5Part1L6helperv()
// CHECK-DAG: define internal {{.*}} @_ZW1AWP5Part2L6helperv()
// CHECK-DAG: call {{.*}} @_ZW1AWP5Part1L6helperv()
// CHECK-DAG: call {{.*}} @_ZW1AWP5Part2L6helperv()
// CHECK-DAG: ret i32 1
// CHECK-DAG: ret i32 1

//--- part1.cppm
module;
static inline __attribute__((noinline)) int helper() { return 1; }
export module A:Part1;
export inline int part1() { return helper(); }

//--- part2.cppm
module;
static inline __attribute__((noinline)) int helper() { return 1; }
export module A:Part2;
export inline int part2() { return helper(); }

//--- A.cppm
export module A;
export import :Part1;
export import :Part2;

//--- use.cpp
import A;
int use() { return part1() + part2(); }
