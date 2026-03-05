// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Template with non-type parameter missing comdat.
//
// CodeGen:
//   $_ZN5ArrayILi5EE4sizeEv = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN5ArrayILi5EE4sizeEv = comdat any

template<int N>
struct Array {
    int data[N];
    int size() const { return N; }
};

int test() {
    Array<5> arr;
    return arr.size();
}
