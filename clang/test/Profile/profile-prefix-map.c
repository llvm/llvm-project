// %s expands to an absolute path, so to test relative paths we need to create a
// clean directory, put the source there, and cd into it.
// RUN: rm -rf %t
// RUN: mkdir -p %t/root/nested
// RUN: echo "void f1() {}" > %t/root/nested/profile-prefix-map.c
// RUN: cd %t/root

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name profile-prefix-map.c %t/root/nested/profile-prefix-map.c -o - | FileCheck --check-prefix=ABSOLUTE %s
//
// ABSOLUTE: @__llvm_coverage_mapping = {{.*"\\02.*root.*nested.*profile-prefix-map\.c}}

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name profile-prefix-map.c ../root/nested/profile-prefix-map.c -o - | FileCheck --check-prefix=RELATIVE %s
//
// RELATIVE: @__llvm_coverage_mapping = {{.*"\\02.*}}..{{/|\\+}}root{{/|\\+}}nested{{.*profile-prefix-map\.c}}

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name profile-prefix-map.c %t/root/nested/profile-prefix-map.c -fprofile-prefix-map=%/t/root=. -o - | FileCheck --check-prefix=PROFILE-PREFIX-MAP %s
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name profile-prefix-map.c ../root/nested/profile-prefix-map.c -fprofile-prefix-map=../root=. -o - | FileCheck --check-prefix=PROFILE-PREFIX-MAP %s
// PROFILE-PREFIX-MAP: @__llvm_coverage_mapping = {{.*"\\02.*}}.{{/|\\+}}nested{{.*profile-prefix-map\.c}}

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name profile-prefix-map.c %t/root/nested/profile-prefix-map.c -fprofile-compilation-dir=/custom -fprofile-prefix-map=/custom=/nonsense -o - | FileCheck --check-prefix=PROFILE-COMPILATION-DIR %s
// PROFILE-COMPILATION-DIR: @__llvm_coverage_mapping = {{.*"\\02.*}}nonsense
