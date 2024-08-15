// RUN: %clang -### --target=x86_64 -c -Wa,--mrelax-relocations=no %s 2>&1 | FileCheck  %s

// CHECK: "-cc1"
// CHECK: "-mrelax-relocations=no"

// RUN: not %clang -### --target=x86_64 -c -Wa,-mrelax-relocations=x %s 2>&1 | FileCheck %s --check-prefix=ERR
// ERR: error: unsupported argument '-mrelax-relocations=x' to option '-Wa,'

// RUN: not %clang -### --target=aarch64 -c -Wa,-mrelax-relocations=no %s 2>&1 | FileCheck %s --check-prefix=ERR2
// ERR2: error: unsupported argument '-mrelax-relocations=no' to option '-Wa,'
