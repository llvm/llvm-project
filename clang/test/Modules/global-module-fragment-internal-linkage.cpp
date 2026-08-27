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
// RUN:   -fmodule-file=%t/A-Part1.pcm -fmodule-file=%t/A-Part2.pcm \
// RUN:   -o %t/A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   %t/use.cpp -fmodule-file=%t/A.pcm \
// RUN:   -fmodule-file=%t/A-Part1.pcm -fmodule-file=%t/A-Part2.pcm \
// RUN:   | FileCheck %s
//
// Two internal functions with the same ordinary mangled name must remain
// distinct when their global module fragments are imported together.
// CHECK-DAG: define internal {{.*}} @_ZLW1AWP5Part16helperv()
// CHECK-DAG: define internal {{.*}} @_ZLW1AWP5Part26helperv()
// CHECK-DAG: call {{.*}} @_ZLW1AWP5Part16helperv()
// CHECK-DAG: call {{.*}} @_ZLW1AWP5Part26helperv()
// CHECK-DAG: ret i32 1
// CHECK-DAG: ret i32 2

//--- part1.cppm
module;
static inline __attribute__((noinline)) int helper() { return 1; }
export module A:Part1;
export inline int part1() { return helper(); }

//--- part2.cppm
module;
static inline __attribute__((noinline)) int helper() { return 2; }
export module A:Part2;
export inline int part2() { return helper(); }

//--- A.cppm
export module A;
export import :Part1;
export import :Part2;

//--- use.cpp
import A;
int use() { return part1() + part2(); }
