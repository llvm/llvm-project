// Check that we error when -faltivec is specified on non-ppc platforms.

// RUN: %clang -target powerpc-unk-unk -faltivec -fsyntax-only %s
// RUN: %clang -target powerpc64-linux-gnu -faltivec -fsyntax-only %s
// RUN: %clang -target powerpc64-linux-gnu -maltivec -fsyntax-only %s

// RUN: %clang -target i386-pc-win32 -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-unknown-freebsd -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -target armv6-apple-darwin -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -target armv7-apple-darwin -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -target mips-linux-gnu -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -target mips64-linux-gnu -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -target sparc-unknown-solaris -faltivec -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: invalid argument '-faltivec' only allowed with 'ppc/ppc64'
