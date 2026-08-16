// Alpine Linux ships its GCC toolchain under a vendor-prefixed musl triple
// (e.g. aarch64-alpine-linux-musl, armv7-alpine-linux-musleabihf). Verify that
// a neutral --target=<arch>-linux-musl[abi] still locates it, matching the
// behavior of the exact Alpine triple.
// See https://github.com/llvm/llvm-project/issues/89146

// RUN: %clang -### %s --target=aarch64-linux-musl --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_aarch64_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=AARCH64 %s
// RUN: %clang -### %s --target=aarch64-alpine-linux-musl --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_aarch64_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=AARCH64 %s
// AARCH64: "{{[^"]*}}/usr/lib/gcc/aarch64-alpine-linux-musl/13.2.1{{/|\\\\}}crtbegin.o"

// RUN: %clang -### %s --target=armv7-linux-musleabihf --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_armv7_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=ARMHF %s
// RUN: %clang -### %s --target=armv7-alpine-linux-musleabihf --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_armv7_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=ARMHF %s
// ARMHF: "{{[^"]*}}/usr/lib/gcc/armv7-alpine-linux-musleabihf/13.2.1{{/|\\\\}}crtbegin.o"

int main(void) {}
