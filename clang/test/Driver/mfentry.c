// RUN: %clang -target s390x -c -### %s -mfentry 2>&1 | FileCheck %s
// RUN: %clang -target i386 -c -### %s -mfentry 2>&1 | FileCheck %s
// RUN: %clang -target x86_64 -c -### %s -mfentry 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-linux-gnu -pg -mfentry -O0 -### -E %s 2>&1 | FileCheck -check-prefix=FP %s
// RUN: %clang -target x86_64-linux-gnu -pg -mfentry -O2 -fno-omit-frame-pointer -### -E %s 2>&1 | FileCheck -check-prefix=FP %s
// RUN: %clang -target x86_64-linux-gnu -pg -mfentry -O2 -### -E %s 2>&1 | FileCheck -check-prefix=NOFP %s
// RUN: %clang -target x86_64 -pg -mfentry -O0 -### -E %s 2>&1 | FileCheck -check-prefix=FP %s
// RUN: %clang -target x86_64 -pg -mfentry -O2 -fno-omit-frame-pointer -### -E %s 2>&1 | FileCheck -check-prefix=FP %s
// RUN: %clang -target x86_64 -pg -mfentry -O2 -### -E %s 2>&1 | FileCheck -check-prefix=FP %s

// CHECK: "-mfentry"

// RUN: %clang -target powerpc64le -c -### %s -mfentry 2>&1 | FileCheck --check-prefix=ERR %s

// ERR: error: unsupported option '-mfentry' for target 'powerpc64le'

// FP: "-mframe-pointer=all"
// NOFP: "-mframe-pointer=none"
void foo(void) {}
