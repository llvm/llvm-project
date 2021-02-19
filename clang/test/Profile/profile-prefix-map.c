// %s expands to an absolute path, so to test relative paths we need to create a
// clean directory, put the source there, and cd into it.
// RUN: rm -rf %t
// RUN: mkdir -p %t/root/nested
// RUN: echo "void f1() {}" > %t/root/nested/profile-prefix-map.c
// RUN: cd %t/root

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name profile-prefix-map.c nested/profile-prefix-map.c -o - | FileCheck --check-prefix=ABSOLUTE %s
//
// ABSOLUTE: @__llvm_coverage_mapping = {{.*"\\01.*root.*nested.*profile-prefix-map\.c}}

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -mllvm -enable-name-compression=false -main-file-name profile-prefix-map.c nested/profile-prefix-map.c -fprofile-prefix-map=%/t/root=. -o - | FileCheck --check-prefix=PROFILE-PREFIX-MAP %s --implicit-check-not=root
//
// PROFILE-PREFIX-MAP: @__llvm_coverage_mapping = {{.*"\\01[^/]*}}.{{/|\\+}}nested{{.*profile-prefix-map\.c}}
