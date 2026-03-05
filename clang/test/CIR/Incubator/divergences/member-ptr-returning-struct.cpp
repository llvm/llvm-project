// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Member function pointer returning struct.
// Both member pointer AND return value have calling convention issues.
//
// CodeGen decomposes member pointer and may coerce return:
//   (check specific signature)
//
// CIR has both issues

// DIFF: Check for both issues

struct Result {
    int value;
};

struct Worker {
    Result compute() { return {42}; }
};

Result call_worker(Worker* w, Result (Worker::*ptr)()) {
    return (w->*ptr)();
}

int test() {
    Worker w;
    return call_worker(&w, &Worker::compute).value;
}
