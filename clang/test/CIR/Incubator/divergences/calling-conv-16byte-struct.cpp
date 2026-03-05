// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// 16-byte structs are at the boundary. On x86_64, they should be coerced to { i64, i64 }.
//
// CodeGen coerces to two i64 registers:
//   define { i64, i64 } @return_four_ints()
//
// CIR returns the struct directly:
//   define %struct.FourInts @return_four_ints()

// DIFF: -define {{.*}} { i64, i64 } @{{.*}}return_four_ints
// DIFF: +define {{.*}} %struct.FourInts @{{.*}}return_four_ints

struct FourInts {
    int a, b, c, d;  // 16 bytes - boundary case
};

FourInts return_four_ints() {
    return {1, 2, 3, 4};
}

int take_four_ints(FourInts s) {
    return s.a + s.b + s.c + s.d;
}

int test() {
    FourInts s = return_four_ints();
    return take_four_ints(s);
}
