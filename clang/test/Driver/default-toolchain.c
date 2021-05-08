// RUN: %clang -target i386-unknown-unknown -m64 -v 2> %t
// RUN: grep 'Target: x86_64-unknown-unknown' %t

// RUN: %clang -target i386-apple-darwin9 -arch ppc -m64 -v 2> %t
// RUN: grep 'Target: powerpc64-apple-darwin9' %t

// RUN: %clang -target i386-apple-darwin9 -arch ppc64 -m32 -v 2> %t
// RUN: grep 'Target: powerpc-apple-darwin9' %t

// RUN: %clang -target x86_64-apple-macos11 -arch arm64 -v 2>&1 | FileCheck --check-prefix=ARM64 %s
// ARM64: Target: arm64-apple-macos11
