// RUN: %clang -### -c --target=x86_64 -march=x86-64 -Wa,-msse2avx %s 2>&1 | FileCheck  %s
// RUN: %clang -### -c --target=x86_64 -march=x86-64 -x assembler -Xassembler -msse2avx %s 2>&1 | FileCheck %s

// CHECK: "-msse2avx"

// RUN: not %clang -### -c --target=aarch64 -march=armv8a -Wa,-msse2avx %s 2>&1 | FileCheck --check-prefix=ERR %s
// ERR:   error: unsupported argument '-msse2avx' to option '-Wa,'
