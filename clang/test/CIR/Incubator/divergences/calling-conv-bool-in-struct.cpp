// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Structs with bool are classified as INTEGER class.
// This 8-byte struct should be coerced to i64.
//
// CodeGen coerces:
//   define i64 @return_bool_struct()
//
// CIR returns struct:
//   define %struct.BoolStruct @return_bool_struct()

// DIFF: -define {{.*}} i64 @{{.*}}return_bool_struct
// DIFF: +define {{.*}} %struct.BoolStruct @{{.*}}return_bool_struct

struct BoolStruct {
    bool b;    // 1 byte
    int x;     // 4 bytes (with padding = 8 bytes total)
};

BoolStruct return_bool_struct() {
    return {true, 42};
}

int test() {
    BoolStruct s = return_bool_struct();
    return s.b ? s.x : 0;
}
