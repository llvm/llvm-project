// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void f(char *fmt, ...);
void test() {
    f("test\0");
}

// LLVM: @.str = {{.*}}[6 x i8] c"test\00\00"
