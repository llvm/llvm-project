// RUN: %clang %s -### --target=arm64-apple-darwin -arch x86_64 -arch arm64 -fuse-lipo=llvm-lipo 2>&1 | FileCheck -check-prefix=TEST1 %s
// TEST1: llvm-lipo

// RUN: %clang %s -### --target=arm64-apple-darwin -arch x86_64 -arch arm64 -fuse-lipo=nonexistant-lipo 2>&1 | FileCheck -check-prefix=TEST2 %s
// TEST2: nonexistant-lipo

// RUN: %clang %s -### --target=arm64-apple-darwin -fuse-lipo=llvm-lipo 2>&1 | FileCheck -check-prefix=TEST3 %s
// TEST3: clang: warning: argument unused during compilation: '-fuse-lipo=llvm-lipo'

// RUN: %clang %s -### --target=arm64-apple-darwin -Wno-unused-command-line-argument -fuse-lipo=llvm-lipo 2>&1 | FileCheck -check-prefix=TEST4 %s
// TEST4-NOT: llvm-lipo

// RUN: %clang %s -### --target=arm64-apple-darwin -arch x86_64 -arch arm64 2>&1 | FileCheck -check-prefix=TEST5 %s
// TEST5: lipo
// TEST5-NOT: llvm-lipo
