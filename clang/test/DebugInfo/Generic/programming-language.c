// RUN: %clang_cc1 -dwarf-version=3 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c -std=c11 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-C11 %s
// RUN: %clang_cc1 -dwarf-version=3 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c -std=c17 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-C17 %s
// RUN: %clang_cc1 -dwarf-version=3 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c -std=c11 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   -gstrict-dwarf | FileCheck --check-prefix=CHECK-STRICT %s
// RUN: %clang_cc1 -dwarf-version=5 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c -std=c11 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   -gstrict-dwarf | FileCheck --check-prefix=CHECK-C11 %s

// CHECK-STRICT: !DICompileUnit(language: DW_LANG_C99
// CHECK-C11: !DICompileUnit(language: DW_LANG_C11
// Update this check once support for DW_LANG_C17 is broadly supported/known in
// consumers. Maybe we'll skip this and go to the DWARFv6 language+version
// encoding that avoids the risk of regression when describing a language
// version newer than what the consumer is aware of.
// CHECK-C17: !DICompileUnit(language: DW_LANG_C11

void f1(void) { }
