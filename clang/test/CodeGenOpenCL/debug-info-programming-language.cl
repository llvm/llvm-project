// RUN: %clang_cc1 -dwarf-version=5 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x cl -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-OPENCL %s
// RUN: %clang_cc1 -dwarf-version=3 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x cl -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-OPENCL %s
// RUN: %clang_cc1 -dwarf-version=3 -gstrict-dwarf -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x cl -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-C99 %s
// RUN: %clang_cc1 -dwarf-version=5 -gstrict-dwarf -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -x cl -O0 -disable-llvm-passes -debug-info-kind=limited \
// RUN:   | FileCheck --check-prefix=CHECK-OPENCL %s

kernel void empty() {}

// CHECK-OPENCL: distinct !DICompileUnit(language: DW_LANG_OpenCL,
// CHECK-C99: distinct !DICompileUnit(language: DW_LANG_C99,
