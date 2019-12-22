// RUN: %clang -target s390x -c -### %s -mfentry 2>&1 | FileCheck %s
// RUN: %clang -target i386 -c -### %s -mfentry 2>&1 | FileCheck %s
// RUN: %clang -target x86_64 -c -### %s -mfentry 2>&1 | FileCheck %s

// CHECK: "-mfentry"

// RUN: %clang -target powerpc64le -c -### %s -mfentry 2>&1 | FileCheck --check-prefix=ERR %s

// ERR: error: unsupported option '-mfentry' for target 'powerpc64le'
