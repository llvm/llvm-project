// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=5 \
// RUN:    | FileCheck %s --implicit-check-not "sourceLanguageName" --implicit-check-not "sourceLanguageVersion"
//
// RUN: %clang_cc1 -emit-llvm %s -o - -debug-info-kind=limited -dwarf-version=6 \
// RUN:    | FileCheck %s --implicit-check-not "sourceLanguageVersion" --check-prefix=CHECK-OBJCXX

int globalVar = 10;

// CHECK-OBJCXX: !DICompileUnit(sourceLanguageName: DW_LNAME_ObjC_plus_plus,
