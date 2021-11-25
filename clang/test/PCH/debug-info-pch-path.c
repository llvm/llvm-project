// Unsupported on AIX because we don't support the requisite "__clangast"
// section in XCOFF yet.
// UNSUPPORTED: aix

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cd %t
//
// ---------------------------------------------------------------------
// Relative PCH, same directory.
// ---------------------------------------------------------------------
//
// RUN: %clang_cc1 -fmodule-format=obj -emit-pch                \
// RUN:     -triple %itanium_abi_triple                         \
// RUN:     -o prefix.pch %S/debug-info-limited-struct.h
//
// RUN: %clang_cc1 -debug-info-kind=standalone                  \
// RUN:     -dwarf-ext-refs -fmodule-format=obj                 \
// RUN:     -triple %itanium_abi_triple                         \
// RUN:     -include-pch prefix.pch %s -emit-llvm -o %t.nodir.ll %s
// RUN: cat %t.nodir.ll | FileCheck %s --check-prefix=CHECK-REL-NODIR
//
//
// CHECK-REL-NODIR: !DICompileUnit
// CHECK-REL-NODIR-SAME:           file: ![[C:[0-9]+]]
// CHECK-REL-NODIR-NOT: dwoId
// CHECK-REL-NODIR: ![[C]] = !DIFile({{.*}}directory: "[[DIR:.*]]"
// CHECK-REL-NODIR: !DICompileUnit(
// CHECK-REL-NODIR-SAME:           file: ![[PCH:[0-9]+]]
// CHECK-REL-NODIR-SAME:           splitDebugFilename: "prefix.pch"
// CHECK-REL-NODIR: ![[PCH]] = !DIFile({{.*}}directory: "[[DIR]]

// ---------------------------------------------------------------------
// Relative PCH in a subdirectory.
// ---------------------------------------------------------------------
//
// RUN: mkdir pchdir
// RUN: %clang_cc1 -fmodule-format=obj -emit-pch                \
// RUN:     -triple %itanium_abi_triple                         \
// RUN:     -o pchdir/prefix.pch %S/debug-info-limited-struct.h
//
// RUN: %clang_cc1 -debug-info-kind=standalone                  \
// RUN:     -dwarf-ext-refs -fmodule-format=obj                 \
// RUN:     -triple %itanium_abi_triple                         \
// RUN:     -include-pch pchdir/prefix.pch %s -emit-llvm -o %t.rel.ll %s
// RUN: cat %t.rel.ll | FileCheck %s --check-prefix=CHECK-REL

// CHECK-REL: !DICompileUnit
// CHECK-REL-SAME:           file: ![[C:[0-9]+]]
// CHECK-REL-NOT: dwoId
// CHECK-REL: ![[C]] = !DIFile({{.*}}directory: "[[DIR:.*]]"
// CHECK-REL: !DICompileUnit(
// CHECK-REL-SAME:           file: ![[PCH:[0-9]+]]
// CHECK-REL-SAME:           splitDebugFilename: "pchdir{{.*}}prefix.pch"
// CHECK-REL: ![[PCH]] = !DIFile({{.*}}directory: "[[DIR]]"

// ---------------------------------------------------------------------
// Absolute PCH.
// ---------------------------------------------------------------------
//
// RUN: %clang_cc1 -fmodule-format=obj -emit-pch                \
// RUN:     -triple %itanium_abi_triple                         \
// RUN:     -o %t/prefix.pch %S/debug-info-limited-struct.h
//
// RUN: %clang_cc1 -debug-info-kind=standalone                  \
// RUN:     -dwarf-ext-refs -fmodule-format=obj                 \
// RUN:     -triple %itanium_abi_triple                         \
// RUN:     -include-pch %t/prefix.pch %s -emit-llvm -o %t.abs.ll %s
// RUN: cat %t.abs.ll | FileCheck %s --check-prefix=CHECK-ABS

// CHECK-ABS: !DICompileUnit
// CHECK-ABS-SAME:           file: ![[C:[0-9]+]]
// CHECK-ABS-NOT: dwoId
// CHECK-ABS: ![[C]] = !DIFile({{.*}}directory: "[[DIR:.*]]"
// CHECK-ABS: !DICompileUnit(
// CHECK-ABS-SAME:           file: ![[PCH:[0-9]+]]
// CHECK-ABS-SAME:           splitDebugFilename: "prefix.pch"
// CHECK-ABS: ![[PCH]] = !DIFile({{.*}}directory: "[[DIR]]
