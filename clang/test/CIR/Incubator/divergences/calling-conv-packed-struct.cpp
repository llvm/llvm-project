// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Packed structs have altered alignment which affects ABI classification.
// This packed struct is 5 bytes and should be coerced differently.
//
// CodeGen may coerce or use sret depending on classification:
//   (check specific LLVM IR)
//
// CIR returns struct:
//   define %struct.PackedStruct @return_packed()

// DIFF: -define {{.*}} @{{.*}}return_packed
// DIFF: +define {{.*}} %struct.PackedStruct @{{.*}}return_packed

struct __attribute__((packed)) PackedStruct {
    char c;    // 1 byte
    int i;     // 4 bytes, no padding - total 5 bytes packed
};

PackedStruct return_packed() {
    return {1, 2};
}

int test() {
    PackedStruct s = return_packed();
    return s.i;
}
