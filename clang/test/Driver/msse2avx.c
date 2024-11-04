// RUN: %clang -### -c -target x86_64 -march=x86-64 -Xassembler -msse2avx %s 2>&1 | FileCheck  %s
// RUN: %clang -### -c -target x86_64 -march=x86-64 -x assembler -Xassembler -msse2avx %s 2>&1 | FileCheck %s

// CHECK: "-msse2avx"

// RUN: not %clang -### -c -target aarch64 -march=armv8a -msse2avx %s 2>&1 | FileCheck --check-prefix=ERR %s
// ERR:   error: unsupported option '-msse2avx' for target 'aarch64'
