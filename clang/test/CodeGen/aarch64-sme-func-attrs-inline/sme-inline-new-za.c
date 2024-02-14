// RUN: %clang --target=aarch64-none-linux-gnu -march=armv9-a+sme -S -Xclang -verify %s

__attribute__((always_inline)) __arm_new("za")
void inline_new_za(void)  { }
// expected-error@+1 {{always_inline function 'inline_new_za' has new za state}}
void inline_caller() { inline_new_za(); }
