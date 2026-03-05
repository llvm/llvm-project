// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Structs containing arrays should follow aggregate ABI rules.
// This struct has 8 bytes (array of 2 ints) and should be coerced to i64.
//
// CodeGen coerces:
//   define i64 @return_array_struct()
//
// CIR returns struct:
//   define %struct.ArrayStruct @return_array_struct()

// DIFF: -define {{.*}} i64 @{{.*}}return_array_struct
// DIFF: +define {{.*}} %struct.ArrayStruct @{{.*}}return_array_struct

struct ArrayStruct {
    int arr[2];  // 8 bytes
};

ArrayStruct return_array_struct() {
    return {{1, 2}};
}

int test() {
    ArrayStruct s = return_array_struct();
    return s.arr[0] + s.arr[1];
}
