// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Structs with long long (8 bytes) should be coerced to i64.
//
// CodeGen coerces:
//   define i64 @return_longlong()
//
// CIR returns struct:
//   define %struct.LongLongStruct @return_longlong()

// DIFF: -define {{.*}} i64 @{{.*}}return_longlong
// DIFF: +define {{.*}} %struct.LongLongStruct @{{.*}}return_longlong

struct LongLongStruct {
    long long ll;  // 8 bytes
};

LongLongStruct return_longlong() {
    return {123456789LL};
}

int test() {
    LongLongStruct s = return_longlong();
    return s.ll > 0 ? 1 : 0;
}
