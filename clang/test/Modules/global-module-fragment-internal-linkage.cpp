// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf -emit-module-interface %t/part1.cppm \
// RUN:   -o %t/A-Part1.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf -emit-module-interface %t/part2.cppm \
// RUN:   -o %t/A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/A-Part2.pcm -o %t/A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fskip-odr-check-in-gmf %t/use.cpp -fmodule-file=%t/A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/A-Part2.pcm | FileCheck %s --check-prefix=SAME
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -DRETURN_VALUE=2 \
// RUN:   -fskip-odr-check-in-gmf -emit-module-interface %t/part2.cppm \
// RUN:   -o %t/different-A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/different-A-Part2.pcm \
// RUN:   -o %t/different-A.pcm
// RUN: not %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fskip-odr-check-in-gmf %t/use.cpp -fmodule-file=%t/different-A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/different-A-Part2.pcm 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT
//
// Equivalent definitions are merged and retain the ordinary internal-linkage
// name.
// SAME-COUNT-2: call {{.*}} @_ZL6helperv()
// SAME-COUNT-1: define internal {{.*}} @_ZL6helperv()

// Non-equivalent definitions remain distinct and are diagnosed through the
// function ODR path, even when general GMF ODR checking is skipped.
// DIFFERENT: error: 'helper' has different definitions in different modules

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
