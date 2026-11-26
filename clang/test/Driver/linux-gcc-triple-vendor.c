// A --target without a vendor finds a GCC installation whose triple differs
// from the target only in the vendor, like Alpine's
// aarch64-alpine-linux-musl, or in the spelling of the architecture, like
// Void's armv7l-linux-musleabihf.

// RUN: %clang -### %s --target=aarch64-linux-musl --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_linux_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=AARCH64 %s
// RUN: %clang -### %s --target=aarch64-unknown-linux-musl --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_linux_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=AARCH64 %s
// AARCH64: "{{.*}}/usr/lib/gcc/aarch64-alpine-linux-musl/14.2.0{{/|\\\\}}crtbegin.o"

/// The sub-architecture has to match.
// RUN: %clang -### %s --target=armv7-linux-musleabihf --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_linux_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=ARMV7 %s
// ARMV7: "{{.*}}/usr/lib/gcc/armv7-alpine-linux-musleabihf/14.2.0{{/|\\\\}}crtbegin.o"
// RUN: %clang -### %s --target=armv6-linux-musleabihf --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_linux_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=ARMV6 %s
// ARMV6: "{{.*}}/usr/lib/gcc/armv6-alpine-linux-musleabihf/14.2.0{{/|\\\\}}crtbegin.o"

// RUN: %clang -### %s --target=armv7-linux-musleabihf --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/void_linux_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=ARMV7L %s
// ARMV7L: "{{.*}}/usr/lib/gcc/armv7l-linux-musleabihf/14.2{{/|\\\\}}crtbegin.o"

/// A known vendor, --gcc-triple, or a different environment disables the match.
// RUN: %clang -### %s --target=aarch64-suse-linux-musl --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_linux_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=NOGCC %s
// RUN: %clang -### %s --target=aarch64-linux-musl --gcc-triple=aarch64-linux-musl \
// RUN:     --rtlib=libgcc -no-pie --sysroot=%S/Inputs/alpine_linux_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=NOGCC %s
// RUN: %clang -### %s --target=aarch64-linux-gnu --rtlib=libgcc -no-pie \
// RUN:     --sysroot=%S/Inputs/alpine_linux_musl_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=NOGCC %s
// NOGCC-NOT: /usr/lib/gcc/

int main(void) {}
