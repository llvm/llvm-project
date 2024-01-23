// RUN: %clang --target=aarch64-none-linux-gnu -march=armv9-a+sme -O3 -S -Xclang -verify %s

__attribute__((always_inline))
void inlined(void) {}

void inline_caller(void) {
    inlined();
}

__arm_new("za")
// expected-error@+2 {{inlining always_inline function 'inlined' into 'inline_caller_new_za' would require a lazy za save}}
void inline_caller_new_za(void) {
    inlined();
}
