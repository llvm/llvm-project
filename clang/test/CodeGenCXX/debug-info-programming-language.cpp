// RUN: %clang_cc1 -dwarf-version=5  -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN: | FileCheck --check-prefix=CHECK-DWARF5 %s
// RUN: %clang_cc1 -dwarf-version=3  -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN: | FileCheck --check-prefix=CHECK-DWARF3 %s

int main() {
  return 0;
}

// CHECK-DWARF5: distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14,
// CHECK-DWARF3: distinct !DICompileUnit(language: DW_LANG_C_plus_plus,
