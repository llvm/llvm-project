// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Structs larger than 16 bytes need sret (structured return via hidden pointer).
//
// CodeGen uses sret:
//   define void @return_five_ints(ptr sret(%struct.FiveInts) %result)
//
// CIR incorrectly returns by value:
//   define %struct.FiveInts @return_five_ints()

// DIFF: -define void @{{.*}}return_five_ints(ptr sret(%struct.FiveInts)
// DIFF: +define {{.*}} %struct.FiveInts @{{.*}}return_five_ints()

struct FiveInts {
    int a, b, c, d, e;  // 20 bytes - over the limit
};

FiveInts return_five_ints() {
    return {1, 2, 3, 4, 5};
}

int test() {
    FiveInts s = return_five_ints();
    return s.a + s.e;
}
