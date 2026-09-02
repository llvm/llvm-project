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
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fskip-odr-check-in-gmf %t/use.cpp -fmodule-file=%t/different-A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/different-A-Part2.pcm \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT
//
// Internal-linkage functions from different global module fragments remain
// distinct entities even when their definitions are equivalent. CodeGen keeps
// the ordinary mangled name and uniquifies only the colliding IR name.
// SAME-DAG: call {{.*}} @_ZL6helperv()
// SAME-DAG: call {{.*}} @_ZL6helperv.[[SAME_SUFFIX:[0-9]+]]()
// SAME-DAG: define internal {{.*}} @_ZL6helperv()
// SAME-DAG: define internal {{.*}} @_ZL6helperv.[[SAME_SUFFIX]]()

// Different definitions also remain distinct and retain their respective
// bodies, even when general GMF ODR checking is skipped.
// DIFFERENT-DAG: call {{.*}} @_ZL6helperv()
// DIFFERENT-DAG: call {{.*}} @_ZL6helperv.[[DIFFERENT_SUFFIX:[0-9]+]]()
// DIFFERENT-DAG: define internal {{.*}} @_ZL6helperv()
// DIFFERENT-DAG: define internal {{.*}} @_ZL6helperv.[[DIFFERENT_SUFFIX]]()
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
