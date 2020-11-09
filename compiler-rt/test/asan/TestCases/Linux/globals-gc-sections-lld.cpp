// RUN: %clangxx_asan %s -o %t -Wl,--gc-sections -fuse-ld=lld -ffunction-sections -fdata-sections -mllvm -asan-globals=0
// RUN: %clangxx_asan %s -o %t -Wl,--gc-sections -fuse-ld=lld -ffunction-sections -fdata-sections -mllvm -asan-globals=1

// https://code.google.com/p/address-sanitizer/issues/detail?id=260
// REQUIRES: lld
// FIXME: This may pass on Android, with non-emulated-tls.
// XFAIL: android
int undefined();

// On i386 clang adds --export-dynamic when linking with ASan, which adds all
// non-hidden globals to GC roots.
__attribute__((visibility("hidden"))) int (*unused)() = undefined;

int main() {
        return 0;
}
