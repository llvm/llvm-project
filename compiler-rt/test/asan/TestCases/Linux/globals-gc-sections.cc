// RUN: %clangxx_asan %s -o %t -Wl,--gc-sections -fuse-ld=bfd -ffunction-sections -fdata-sections -mllvm -asan-globals=0
// RUN: %clangxx_asan %s -o %t -Wl,--gc-sections -fuse-ld=bfd -ffunction-sections -fdata-sections -mllvm -asan-globals=1

// https://code.google.com/p/address-sanitizer/issues/detail?id=260

int undefined();

// bug in ld.bfd: with multiple "asan_globals" sections, __start_asan_globals is
// treated as a strong GC reference to the first such section. As a result, the
// first (for some definition of the word) global is never gc-ed.
int first_unused = 42;

// On i386 clang adds --export-dynamic when linking with ASan, which adds all
// non-hidden globals to GC roots.
__attribute__((visibility("hidden"))) int (*unused)() = undefined;

int main() {
        return 0;
}
