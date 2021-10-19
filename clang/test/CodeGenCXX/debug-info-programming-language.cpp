// RUN: %clang_cc1 -dwarf-version=5 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-CPP14 %s
// RUN: %clang_cc1 -dwarf-version=3 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-CPP14 %s
// RUN: %clang_cc1 -dwarf-version=3 -gstrict-dwarf -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited | FileCheck %s
// RUN: %clang_cc1 -dwarf-version=5 -gstrict-dwarf -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-CPP14 %s

int main() {
  return 0;
}

// CHECK-CPP14: distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14,
// CHECK: distinct !DICompileUnit(language: DW_LANG_C_plus_plus,
