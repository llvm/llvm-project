// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   -fmodules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/part1.cppm -o %t/A-Part1.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   -fmodules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/part2.cppm -o %t/A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   -fmodules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/A-Part2.pcm \
// RUN:   -o %t/A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fno-skip-odr-check-in-gmf \
// RUN:   -fmodules-unique-gmf-internal-linkage \
// RUN:   %t/use.cpp -fmodule-file=%t/A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/A-Part2.pcm \
// RUN:   | FileCheck %s
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf \
// RUN:   -fno-modules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/part1.cppm -o %t/no-A-Part1.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf \
// RUN:   -fno-modules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/part2.cppm -o %t/no-A-Part2.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 \
// RUN:   -fskip-odr-check-in-gmf \
// RUN:   -fno-modules-unique-gmf-internal-linkage \
// RUN:   -emit-module-interface %t/A.cppm \
// RUN:   -fmodule-file=A:Part1=%t/no-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/no-A-Part2.pcm \
// RUN:   -o %t/no-A.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -emit-llvm -o - \
// RUN:   -fskip-odr-check-in-gmf \
// RUN:   -fno-modules-unique-gmf-internal-linkage \
// RUN:   %t/use.cpp -fmodule-file=%t/no-A.pcm \
// RUN:   -fmodule-file=A:Part1=%t/no-A-Part1.pcm \
// RUN:   -fmodule-file=A:Part2=%t/no-A-Part2.pcm \
// RUN:   | FileCheck %s --check-prefix=NO-DISAMBIGUATION
//
// Identical internal functions from the same textual header must remain
// distinct when separate global module fragments are imported together.
// CHECK-DAG: define internal {{.*}} @_ZW1AWP5Part1L6helperv()
// CHECK-DAG: define internal {{.*}} @_ZW1AWP5Part2L6helperv()
// CHECK-DAG: call {{.*}} @_ZW1AWP5Part1L6helperv()
// CHECK-DAG: call {{.*}} @_ZW1AWP5Part2L6helperv()
// CHECK-DAG: ret i32 1
// CHECK-DAG: ret i32 1

// NO-DISAMBIGUATION: define internal {{.*}} @_ZL6helperv()
// NO-DISAMBIGUATION-NOT: @_ZW1AWP5Part

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
