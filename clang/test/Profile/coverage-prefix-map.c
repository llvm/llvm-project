// %s expands to an absolute path, so to test relative paths we need to create a
// clean directory, put the source there, and cd into it.
// RUN: rm -rf %t
// RUN: mkdir -p %t/root/nested
// RUN: echo "void f1(void) {}" > %t/root/nested/coverage-prefix-map.c
// RUN: cd %t/root

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name coverage-prefix-map.c %t/root/nested/coverage-prefix-map.c -o - | FileCheck --check-prefix=ABSOLUTE %s
//
// ABSOLUTE: @__llvm_coverage_mapping = {{.*"\\02.*root.*nested.*coverage-prefix-map\.c}}

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name coverage-prefix-map.c ../root/nested/coverage-prefix-map.c -o - | FileCheck --check-prefix=RELATIVE %s
//
// RELATIVE: @__llvm_coverage_mapping = {{.*"\\02.*}}..{{/|\\+}}root{{/|\\+}}nested{{.*coverage-prefix-map\.c}}

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name coverage-prefix-map.c %t/root/nested/coverage-prefix-map.c -fcoverage-prefix-map=%/t/root=. -o - | FileCheck --check-prefix=COVERAGE-PREFIX-MAP %s
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name coverage-prefix-map.c ../root/nested/coverage-prefix-map.c -fcoverage-prefix-map=../root=. -o - | FileCheck --check-prefix=COVERAGE-PREFIX-MAP %s
// COVERAGE-PREFIX-MAP: @__llvm_coverage_mapping = {{.*"\\02.*}}.{{/|\\+}}nested{{.*coverage-prefix-map\.c}}

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name coverage-prefix-map.c %t/root/nested/coverage-prefix-map.c -fcoverage-compilation-dir=/custom -fcoverage-prefix-map=/custom=/nonsense -o - | FileCheck --check-prefix=COVERAGE-COMPILATION-DIR %s
// COVERAGE-COMPILATION-DIR: @__llvm_coverage_mapping = {{.*"\\02.*}}nonsense

// Test that last -fcoverage-prefix-map option (-fcoverage-prefix-map==newpath) is applied.
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name coverage-prefix-map.c %t/root/nested/coverage-prefix-map.c -fcoverage-prefix-map=%/t/root=. -fcoverage-prefix-map==newpath -o - | FileCheck --check-prefix=COVERAGE-PREFIX-MAP-ORDER %s
// COVERAGE-PREFIX-MAP-ORDER: @__llvm_coverage_mapping = {{.*"\\02.*newpath.*root.*nested.*coverage-prefix-map\.c}}

// Test that last -fcoverage-prefix-map option (-fcoverage-prefix-map=%/t/root=.) is applied.
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name coverage-prefix-map.c %t/root/nested/coverage-prefix-map.c -fcoverage-prefix-map==newpath -fcoverage-prefix-map=%/t/root=. -o - | FileCheck --check-prefix=COVERAGE-PREFIX-MAP-REORDER %s
// COVERAGE-PREFIX-MAP-REORDER: @__llvm_coverage_mapping =
// COVERAGE-PREFIX-MAP-REORDER-NOT: newpath
// COVERAGE-PREFIX-MAP-REORDER-SAME: nested{{.*coverage-prefix-map\.c}}
