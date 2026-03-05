// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Multiple struct parameters should each follow ABI rules.
// Two 8-byte structs should each be coerced to i64.
//
// CodeGen coerces both:
//   define i32 @take_two_structs(i64 %p1.coerce, i64 %p2.coerce)
//
// CIR passes structs directly:
//   define i32 @take_two_structs(%struct.Pair %p1, %struct.Pair %p2)

// DIFF: -define {{.*}} @{{.*}}take_two_structs(i64{{.*}}, i64
// DIFF: +define {{.*}} @{{.*}}take_two_structs(%struct.Pair{{.*}}, %struct.Pair

struct Pair {
    int a, b;  // 8 bytes
};

int take_two_structs(Pair p1, Pair p2) {
    return p1.a + p2.b;
}

int test() {
    return take_two_structs({1, 2}, {3, 4});
}
