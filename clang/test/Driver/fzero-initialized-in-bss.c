// RUN: %clang -### %s -c -fzero-initialized-in-bss 2>&1 | FileCheck %s --check-prefix=NO
// RUN: %clang -### %s -c 2>&1 | FileCheck %s --check-prefix=NO

// NO-NOT: -fno-zero-initialized-in-bss

// RUN: %clang -### %s -c -fzero-initialized-in-bss -fno-zero-initialized-in-bss 2>&1 | FileCheck %s

// CHECK: -fno-zero-initialized-in-bss
