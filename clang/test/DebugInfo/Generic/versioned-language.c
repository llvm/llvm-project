// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=5 -std=c99 \
// RUN:    | FileCheck %s --implicit-check-not "sourceLanguageName" --implicit-check-not "sourceLanguageVersion"
//
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c89 | FileCheck %s --check-prefix=CHECK-C89 --implicit-check-not "sourceLanguageVersion"
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c99 | FileCheck %s --check-prefix=CHECK-C99
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c17 | FileCheck %s --check-prefix=CHECK-C17
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c23 | FileCheck %s --check-prefix=CHECK-C23
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c2y | FileCheck %s --check-prefix=CHECK-C2Y

int globalVar = 10;

// CHECK-C89: !DICompileUnit(sourceLanguageName: DW_LNAME_C,
// CHECK-C99: !DICompileUnit(sourceLanguageName: DW_LNAME_C, sourceLanguageVersion: 199901
// CHECK-C11: !DICompileUnit(sourceLanguageName: DW_LNAME_C, sourceLanguageVersion: 201112
// CHECK-C17: !DICompileUnit(sourceLanguageName: DW_LNAME_C, sourceLanguageVersion: 201710
// CHECK-C23: !DICompileUnit(sourceLanguageName: DW_LNAME_C, sourceLanguageVersion: 202311
// CHECK-C2Y: !DICompileUnit(sourceLanguageName: DW_LNAME_C, sourceLanguageVersion: 202400
