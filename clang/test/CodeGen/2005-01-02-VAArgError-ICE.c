// This file is erroneous, but should not cause the compiler to ICE.
// PR481
// RUN: %clang_cc1 %s -Wno-implicit-function-declaration -emit-llvm -o /dev/null

int flags(int a, int b, ...) {
        __builtin_va_list         args;
        __builtin_va_start(args,a);       // not the last named arg
        foo(args);
}
