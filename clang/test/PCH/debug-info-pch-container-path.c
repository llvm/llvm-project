// REQUIRES: asserts

// Modules:
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cd %t

// RUN: %clang_cc1 -fmodule-format=obj -emit-pch                \
// RUN:     -triple %itanium_abi_triple                         \
// RUN:     -fdebug-prefix-map=%t=BUILD                         \
// RUN:     -fdebug-prefix-map=%S=SOURCE                        \
// RUN:     -o %t/prefix.ll %S/debug-info-limited-struct.h      \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-container.ll
// RUN: cat %t-container.ll | FileCheck %s

// CHECK: distinct !DICompileUnit(
// CHECK-SAME:                    language: DW_LANG_C99,
// CHECK-SAME:                    file: ![[FILE:[0-9]+]],
// CHECK: ![[FILE]] = !DIFile(
// CHECK-SAME:                filename: "SOURCE/debug-info-limited-struct.h",
// CHECK-SAME:                directory: "BUILD"

