// RUN: %clang_cc1  -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -mllvm -profile-correlate=binary -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -o - %s | FileCheck %s --check-prefix=BIN-CORRELATE

// CHECK: @__llvm_profile_raw_version = {{.*}} i64 8
// BIN-CORRELATE: @__llvm_profile_raw_version = {{.*}} i64 4294967304

int main() {
    return 0;
}
