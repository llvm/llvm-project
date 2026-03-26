// RUN: %clang_cc1 -dwarf-version=5 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-CPP14 %s
// RUN: %clang_cc1 -dwarf-version=3 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-CPP14 %s
// RUN: %clang_cc1 -dwarf-version=3 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++17 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-CPP17 %s
// RUN: %clang_cc1 -dwarf-version=3 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++20 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-CPP20 %s
// RUN: %clang_cc1 -dwarf-version=3 -gstrict-dwarf -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited | FileCheck %s
// RUN: %clang_cc1 -dwarf-version=5 -gstrict-dwarf -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x c++ -std=c++14 -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-CPP14 %s

int main() {
  return 0;
}

// Update these tests once support for DW_LANG_C_plus_plus_17/20 is added - it's
// a complicated tradeoff. The language codes are already published/blessed by
// the DWARF committee, but haven't been released in a published standard yet,
// so consumers might not be ready for these codes & could regress functionality
// (because they wouldn't be able to identify that the language was C++). The
// DWARFv6 language encoding, separating language from language version, would
// remove this problem/not require new codes for new language versions and make
// it possible to identify the base language irrespective of the version.
// CHECK-CPP14: distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14,
// CHECK-CPP17: distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14,
// CHECK-CPP20: distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14,
// CHECK: distinct !DICompileUnit(language: DW_LANG_C_plus_plus,
