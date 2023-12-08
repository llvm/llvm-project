/// PR47391: if the filename is absolute and starts with current working
/// directory, there may be two ways describing the filename field of DIFile.
/// Test that we canonicalize the DIFile.
// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: cp %s .
// RUN: %clang_cc1 -triple %itanium_abi_triple -main-file-name difile_entry.cpp -debug-info-kind=limited %t/difile_entry.cpp -std=c++11 -emit-llvm -o - | FileCheck %s
int x();
static int i = x();

// CHECK: distinct !DIGlobalVariable(name: "i", {{.*}}, file: ![[#FILE:]],
// CHECK: ![[#FILE]] = !DIFile(filename: "difile_entry.cpp", directory:
// CHECK: distinct !DISubprogram(name: "__cxx_global_var_init", {{.*}}, file: ![[#FILE]],
// CHECK: distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_difile_entry.cpp", {{.*}}, file: ![[#FILE]]
