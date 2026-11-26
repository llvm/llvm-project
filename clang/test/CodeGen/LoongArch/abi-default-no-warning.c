// REQUIRES: loongarch-registered-target

// The default ABI the driver picks for a triple without an environment is one
// that the default features support, so the backend does not warn about it.
// RUN: %clang --target=loongarch32-unknown-none-elf -S -o /dev/null %s 2>&1 | count 0
// RUN: %clang --target=loongarch64-unknown-none-elf -S -o /dev/null %s 2>&1 | count 0

float f(float a, float b) { return a + b; }
