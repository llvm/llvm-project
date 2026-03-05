// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Nested structs should follow the same ABI rules as flat structs.
// This 16-byte nested struct should be coerced to { i64, i64 }.
//
// CodeGen coerces:
//   define { i64, i64 } @return_nested()
//
// CIR returns struct directly:
//   define %struct.Outer @return_nested()

// DIFF: -define {{.*}} { i64, i64 } @{{.*}}return_nested
// DIFF: +define {{.*}} %struct.Outer @{{.*}}return_nested

struct Inner {
    int x, y;  // 8 bytes
};

struct Outer {
    Inner i1, i2;  // 16 bytes total
};

Outer return_nested() {
    return {{1, 2}, {3, 4}};
}

int test() {
    Outer o = return_nested();
    return o.i1.x + o.i2.y;
}
