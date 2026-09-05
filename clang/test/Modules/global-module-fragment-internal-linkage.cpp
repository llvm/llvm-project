// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// The dedicated option is enabled by default and is independent of GMF ODR
// checking.
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf -emit-module-interface %t/part1.cppm \
// RUN:   -o %t/A-Part1.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf -emit-module-interface %t/part2.cppm \
// RUN:   -o %t/A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/A-Part2.pcm -o %t/A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fno-skip-odr-check-in-gmf %t/use.cpp -fmodule-file=%t/A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/A-Part2.pcm \
// RUN:   | FileCheck %s --check-prefix=DEFAULT
//
// Different bodies also remain distinct with the option explicitly enabled.
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -DRETURN_VALUE=2 \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   -fmodules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/part2.cppm -o %t/different-A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   -fmodules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/different-A-Part2.pcm \
// RUN:   -o %t/different-A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   -fmodules-unique-gmf-internal-linkage \
// RUN:   %t/use.cpp -fmodule-file=%t/different-A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/different-A-Part2.pcm \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT
//
// Disabling the option restores the ordinary internal-linkage identity and
// mangling, even when GMF ODR checking is skipped.
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf \
// RUN:   -fno-modules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/part1.cppm -o %t/ordinary-A-Part1.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf \
// RUN:   -fno-modules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/part2.cppm -o %t/ordinary-A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf \
// RUN:   -fno-modules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/ordinary-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/ordinary-A-Part2.pcm \
// RUN:   -o %t/ordinary-A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fskip-odr-check-in-gmf \
// RUN:   -fno-modules-unique-gmf-internal-linkage \
// RUN:   %t/use.cpp -fmodule-file=%t/ordinary-A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/ordinary-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/ordinary-A-Part2.pcm \
// RUN:   | FileCheck %s --check-prefix=ORDINARY
//
// DEFAULT-DAG: call {{.*}} @_ZW1AWP5Part1L6helperv()
// DEFAULT-DAG: call {{.*}} @_ZW1AWP5Part2L6helperv()
// DEFAULT-DAG: define internal {{.*}} @_ZW1AWP5Part1L6helperv()
// DEFAULT-DAG: define internal {{.*}} @_ZW1AWP5Part2L6helperv()
// DEFAULT-DAG: ret i32 1
// DEFAULT-DAG: ret i32 1

// DIFFERENT-DAG: call {{.*}} @_ZW1AWP5Part1L6helperv()
// DIFFERENT-DAG: call {{.*}} @_ZW1AWP5Part2L6helperv()
// DIFFERENT-DAG: define internal {{.*}} @_ZW1AWP5Part1L6helperv()
// DIFFERENT-DAG: define internal {{.*}} @_ZW1AWP5Part2L6helperv()
// DIFFERENT-DAG: ret i32 1
// DIFFERENT-DAG: ret i32 2

// ORDINARY: define internal {{.*}} @_ZL6helperv()
// ORDINARY-NOT: @_ZW1AWP5Part

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
