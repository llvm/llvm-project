// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=5 -std=c++98 \
// RUN:    | FileCheck %s --implicit-check-not "sourceLanguageName" --implicit-check-not "sourceLanguageVersion"
//
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c++98 | FileCheck %s --check-prefix=CHECK-CPP98
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c++03 | FileCheck %s --check-prefix=CHECK-CPP03
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c++11 | FileCheck %s --check-prefix=CHECK-CPP11
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c++14 | FileCheck %s --check-prefix=CHECK-CPP14
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c++17 | FileCheck %s --check-prefix=CHECK-CPP17
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c++20 | FileCheck %s --check-prefix=CHECK-CPP20
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c++23 | FileCheck %s --check-prefix=CHECK-CPP23
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 -std=c++2c | FileCheck %s --check-prefix=CHECK-CPP2C

struct Foo {} globalVar;

// CHECK-CPP98:     !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 199711
// FIXME: C++03 technically has no official standard version code. From Clang's point of view C++03 and C++98 are interchangable.
// CHECK-CPP03:     !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 199711
// CHECK-CPP11:     !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 201103
// CHECK-CPP14:     !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 201402
// CHECK-CPP17:     !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 201703
// CHECK-CPP20:     !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 202002
// CHECK-CPP23:     !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 202302
// CHECK-CPP2C:     !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 202400
