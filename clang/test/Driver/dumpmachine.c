/// Test that -dumpmachine prints the target triple.

/// Note: Debian GCC may omit "unknown-".
// RUN: %clang --target=x86_64-linux-gnu -dumpmachine | FileCheck %s --check-prefix=X86_64
// X86_64: x86_64-unknown-linux-gnu

/// Note: GCC doesn't convert -dumpmachine output for multilib -m32/-mx32/-m64.
// RUN: %clang --target=x86_64-redhat-linux -m32 -dumpmachine | FileCheck %s --check-prefix=X86_64_M32
// X86_64_M32: i386-redhat-linux

// RUN: %clang --target=xxx-pc-freebsd -dumpmachine | FileCheck %s --check-prefix=FREEBSD
// FREEBSD: xxx-pc-freebsd
