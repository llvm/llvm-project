// RUN: %clang -### -c -fc++-static-destructors=all %s 2>&1 | FileCheck --check-prefix ALL %s
// RUN: %clang -### -c -fc++-static-destructors %s 2>&1 | FileCheck --check-prefix ALL %s
// RUN: %clang -### -c -fno-c++-static-destructors -fc++-static-destructors %s 2>&1 | FileCheck --check-prefix ALL %s
// RUN: %clang -### -c -fc++-static-destructors=none %s 2>&1 | FileCheck --check-prefix NONE %s
// RUN: %clang -### -c -fno-c++-static-destructors %s 2>&1 | FileCheck --check-prefix NONE %s
// RUN: %clang -### -c -fc++-static-destructors -fno-c++-static-destructors %s 2>&1 | FileCheck --check-prefix NONE %s
// RUN: %clang -### -c -fc++-static-destructors=thread-local %s 2>&1 | FileCheck --check-prefix THREAD-LOCAL %s

// ALL: -fc++-static-destructors=all
// NONE: -fc++-static-destructors=none
// THREAD-LOCAL: -fc++-static-destructors=thread-local
