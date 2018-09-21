// RUN: %clang -target i386-apple-darwin9 -m32 -Xarch_i386 -O2 %s -S -### 2>&1 | FileCheck -check-prefix=O2ONCE %s
// O2ONCE: "-O2"
// O2ONCE-NOT: "-O2"

// RUN: %clang -target i386-apple-darwin9 -m64 -Xarch_i386 -O2 %s -S -### 2>&1 | FileCheck -check-prefix=O2NONE %s
// O2NONE-NOT: "-O2"
// O2NONE: argument unused during compilation: '-Xarch_i386 -O2'

// RUN: not %clang -target i386-apple-darwin9 -m32 -Xarch_i386 -o -Xarch_i386 -S %s -S -Xarch_i386 -o 2>&1 | FileCheck -check-prefix=INVALID %s
// INVALID: error: invalid Xarch argument: '-Xarch_i386 -o'
// INVALID: error: invalid Xarch argument: '-Xarch_i386 -S'
// INVALID: error: invalid Xarch argument: '-Xarch_i386 -o'
