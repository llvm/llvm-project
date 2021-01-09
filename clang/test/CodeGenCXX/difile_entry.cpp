// RUN: rm -rf %t/test_dir
// RUN: mkdir -p %t/test_dir
// RUN: cd %t/test_dir
// RUN: cp %s .
// RUN: %clang_cc1 -main-file-name difile_entry.cpp  -debug-info-kind=limited ../test_dir/difile_entry.cpp -std=c++11 -emit-llvm -o - | FileCheck  ../test_dir/difile_entry.cpp
int x();
static int i = x();

// CHECK: [[FILE: *]] = !DIFile(filename: "{{.*}}difile_entry.cpp",
// CHECK: {{.*}} = distinct !DISubprogram(name: "__cxx_global_var_init", scope: {{.*}}, file: [[FILE]]
// CHECK: {{.*}} = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_difile_entry.cpp", scope: {{.*}}, file: [[FILE]]

