// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// Identical and different bodies remain distinct with the default GMF ODR
// checking behavior.
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -emit-module-interface %t/part1.cppm -o %t/default-A-Part1.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -emit-module-interface %t/part2.cppm -o %t/default-A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/default-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/default-A-Part2.pcm -o %t/default-A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   %t/use.cpp -fmodule-file=%t/default-A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/default-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/default-A-Part2.pcm \
// RUN:   | FileCheck %s --check-prefix=IDENTICAL
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -DRETURN_VALUE=2 \
// RUN:   -emit-module-interface %t/part2.cppm -o %t/default-different-A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/default-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/default-different-A-Part2.pcm \
// RUN:   -o %t/default-different-A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   %t/use.cpp -fmodule-file=%t/default-different-A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/default-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/default-different-A-Part2.pcm \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT
//
// Repeat both cases with GMF ODR checking enabled.
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf -emit-module-interface %t/part1.cppm \
// RUN:   -o %t/odr-A-Part1.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf -emit-module-interface %t/part2.cppm \
// RUN:   -o %t/odr-A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/odr-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/odr-A-Part2.pcm -o %t/odr-A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fno-skip-odr-check-in-gmf %t/use.cpp -fmodule-file=%t/odr-A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/odr-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/odr-A-Part2.pcm \
// RUN:   | FileCheck %s --check-prefix=IDENTICAL
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -DRETURN_VALUE=2 \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   -emit-module-interface %t/part2.cppm -o %t/odr-different-A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/odr-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/odr-different-A-Part2.pcm \
// RUN:   -o %t/odr-different-A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   %t/use.cpp -fmodule-file=%t/odr-different-A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/odr-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/odr-different-A-Part2.pcm \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT
//
// IDENTICAL-DAG: call {{.*}} @_ZW1AWP5Part1L6helperv()
// IDENTICAL-DAG: call {{.*}} @_ZW1AWP5Part2L6helperv()
// IDENTICAL-DAG: define internal {{.*}} @_ZW1AWP5Part1L6helperv()
// IDENTICAL-DAG: define internal {{.*}} @_ZW1AWP5Part2L6helperv()
// IDENTICAL-DAG: ret i32 1
// IDENTICAL-DAG: ret i32 1

// DIFFERENT-DAG: call {{.*}} @_ZW1AWP5Part1L6helperv()
// DIFFERENT-DAG: call {{.*}} @_ZW1AWP5Part2L6helperv()
// DIFFERENT-DAG: define internal {{.*}} @_ZW1AWP5Part1L6helperv()
// DIFFERENT-DAG: define internal {{.*}} @_ZW1AWP5Part2L6helperv()
// DIFFERENT-DAG: ret i32 1
// DIFFERENT-DAG: ret i32 2

//--- part1.cppm
module;
extern "C" {
static inline __attribute__((noinline)) int helper() { return 1; }
}
export module A:Part1;
export inline int part1() { return helper(); }

//--- part2.cppm
module;
#ifndef RETURN_VALUE
#define RETURN_VALUE 1
#endif
extern "C" {
static inline __attribute__((noinline)) int helper() { return RETURN_VALUE; }
}
export module A:Part2;
export inline int part2() { return helper(); }

//--- A.cppm
export module A;
export import :Part1;
export import :Part2;

//--- use.cpp
import A;
int use() { return part1() + part2(); }
