// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// 12-byte structs (three ints) should be coerced to { i64, i32 } per x86_64 ABI.
//
// CodeGen coerces to two registers:
//   define { i64, i32 } @return_three_ints()
//
// CIR returns the struct directly:
//   define %struct.ThreeInts @return_three_ints()

// DIFF: -define {{.*}} { i64, i32 } @{{.*}}return_three_ints
// DIFF: +define {{.*}} %struct.ThreeInts @{{.*}}return_three_ints

struct ThreeInts {
    int a, b, c;  // 12 bytes total
};

ThreeInts return_three_ints() {
    return {1, 2, 3};
}

int take_three_ints(ThreeInts s) {
    return s.a + s.b + s.c;
}

int test() {
    ThreeInts s = return_three_ints();
    return take_three_ints(s);
}
