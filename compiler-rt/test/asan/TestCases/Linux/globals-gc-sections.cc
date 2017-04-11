// RUN: %clangxx_asan %s -o %t -Wl,--gc-sections -ffunction-sections -mllvm -asan-globals=0
// RUN: %clangxx_asan %s -o %t -Wl,--gc-sections -ffunction-sections -mllvm -asan-globals=1

// https://code.google.com/p/address-sanitizer/issues/detail?id=260
// XFAIL: *

int undefined();

int (*unused)() = undefined;

int main() {
        return 0;
}
