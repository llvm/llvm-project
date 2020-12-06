/// Without instrumenting globals, --gc-sections drops the undefined symbol.
// RUN: %clangxx_asan %s -o /dev/null -Wl,--gc-sections -fuse-ld=lld -ffunction-sections -fdata-sections -mllvm -asan-globals=0
/// With -fsanitize-address-globals-dead-stripping and -fdata-sections, a garbage
/// collectable custom metadata section is used for instrumented globals.
// RUN: %clangxx_asan %s -o /dev/null -Wl,--gc-sections -fuse-ld=lld -ffunction-sections -fdata-sections -fsanitize-address-globals-dead-stripping

// https://github.com/google/sanitizers/issues/260
// REQUIRES: lld-available
int undefined();

// On i386 clang adds --export-dynamic when linking with ASan, which adds all
// non-hidden globals to GC roots.
__attribute__((visibility("hidden"))) int (*unused)() = undefined;

int main() {
        return 0;
}
