// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// 4-byte structs should be coerced to i32 per x86_64 ABI.
//
// CodeGen correctly coerces to i32:
//   define i32 @return_tiny()
//
// CIR incorrectly returns the struct:
//   define %struct.TinyStruct @return_tiny()

// DIFF: -define {{.*}} i32 @{{.*}}return_tiny
// DIFF: +define {{.*}} %struct.TinyStruct @{{.*}}return_tiny

struct TinyStruct {
    int x;  // 4 bytes
};

TinyStruct return_tiny() {
    return {42};
}

int take_tiny(TinyStruct s) {
    return s.x;
}

int test() {
    TinyStruct s = return_tiny();
    return take_tiny(s);
}
