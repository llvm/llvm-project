// RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm %s | FileCheck %s

// Mangle argc/argv main even when it's not defined in this TU.

#include <stddef.h>

int main(int argc, char *argv[]);

int foo(void) {
    return main(0, NULL);
}

// CHECK: call i32 @__main_argc_argv(
