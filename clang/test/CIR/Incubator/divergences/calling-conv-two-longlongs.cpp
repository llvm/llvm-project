// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Two long longs (16 bytes total) should be coerced to { i64, i64 }.
//
// CodeGen coerces:
//   define { i64, i64 } @return_two_longlongs()
//
// CIR returns struct:
//   define %struct.TwoLongLongs @return_two_longlongs()

// DIFF: -define {{.*}} { i64, i64 } @{{.*}}return_two_longlongs
// DIFF: +define {{.*}} %struct.TwoLongLongs @{{.*}}return_two_longlongs

struct TwoLongLongs {
    long long a, b;  // 16 bytes
};

TwoLongLongs return_two_longlongs() {
    return {123LL, 456LL};
}

int test() {
    TwoLongLongs s = return_two_longlongs();
    return s.a + s.b > 0 ? 1 : 0;
}
