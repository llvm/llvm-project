// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Structs containing pointers (8 bytes on x86_64) should be coerced to i64.
//
// CodeGen coerces:
//   define i64 @return_ptr_struct()
//
// CIR returns struct:
//   define %struct.PtrStruct @return_ptr_struct()

// DIFF: -define {{.*}} i64 @{{.*}}return_ptr_struct
// DIFF: +define {{.*}} %struct.PtrStruct @{{.*}}return_ptr_struct

struct PtrStruct {
    int* ptr;  // 8 bytes on x86_64
};

PtrStruct return_ptr_struct() {
    static int x = 42;
    return {&x};
}

int test() {
    PtrStruct s = return_ptr_struct();
    return *s.ptr;
}
