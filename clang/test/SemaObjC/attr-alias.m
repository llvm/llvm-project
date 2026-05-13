// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fblocks -verify -emit-llvm-only %s

// Compiler-generated functions are not valid alias targets.
void foo() { void(^myBlock)(void) = ^{ }; }
void bar() __attribute__((alias("__foo_block_invoke")));
// expected-error@-1 {{alias must point to a defined variable or function}}
