// RUN: %clang -target i386-apple-darwin11 -m32 -Xarch_i386 -O3 %s -S -### 2>&1 | FileCheck -check-prefix=O3ONCE %s
// O3ONCE: "-O3"
// O3ONCE-NOT: "-O3"

// RUN: %clang -target i386-apple-darwin11 -m64 -Xarch_i386 -O3 %s -S -### 2>&1 | FileCheck -check-prefix=O3NONE %s
// O3NONE-NOT: "-O3"
// O3NONE: argument unused during compilation: '-Xarch_i386 -O3'

// RUN: not %clang -target i386-apple-darwin11 -m32 -Xarch_i386 -o -Xarch_i386 -S %s -S -Xarch_i386 -o 2>&1 | FileCheck -check-prefix=INVALID %s
// INVALID: error: invalid Xarch argument: '-Xarch_i386 -o'
// INVALID: error: invalid Xarch argument: '-Xarch_i386 -S'
// INVALID: error: invalid Xarch argument: '-Xarch_i386 -o'
