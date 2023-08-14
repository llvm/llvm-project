// RUN: %clang_cc1 -debug-info-kind=standalone -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=standalone -mllvm -debug-info-correlate -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -o - %s | FileCheck %s --check-prefix=DEBUG_INFO

// CHECK: @__llvm_profile_raw_version = hidden constant i64 8, comdat
// DEBUG_INFO: @__llvm_profile_raw_version = hidden constant i64 576460752303423496, comdat

int main() {
    return 0;
}
