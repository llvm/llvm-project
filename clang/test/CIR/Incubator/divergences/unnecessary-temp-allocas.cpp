// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// CIR generates unnecessary temporary allocas for return values.
// This is a code quality issue - the IR is more verbose than necessary.
//
// CodeGen directly returns values:
//   %call = call i32 @foo()
//   ret i32 %call
//
// CIR allocates a temporary, stores to it, then loads from it:
//   %1 = alloca i32, i64 1, align 4
//   %2 = call i32 @foo()
//   store i32 %2, ptr %1, align 4
//   %3 = load i32, ptr %1, align 4
//   ret i32 %3
//
// This pattern appears in nearly all functions.
// Impact: More verbose IR, extra instructions
// Likely optimized away by later passes, but unnecessary

// DIFF: -  %call = call {{.*}} @_Z3foov()
// DIFF: -  ret i32 %call
// DIFF: +  %{{[0-9]+}} = alloca i32
// DIFF: +  %{{[0-9]+}} = call {{.*}} @_Z3foov()
// DIFF: +  store i32 %{{[0-9]+}}, ptr %{{[0-9]+}}
// DIFF: +  %{{[0-9]+}} = load i32, ptr %{{[0-9]+}}
// DIFF: +  ret i32 %{{[0-9]+}}

int foo() {
    return 42;
}

int test() {
    return foo();
}

// Also affects struct returns
struct S {
    int x, y;
};

S bar() {
    return {1, 2};
}

S test_struct() {
    return bar();
}

// And void functions with calls
void baz() {}

void test_void() {
    baz();
}
