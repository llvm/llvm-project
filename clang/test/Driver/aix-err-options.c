// RUN: %clang --target=powerpc-ibm-aix-xcoff -### -E -G 0 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECK32 %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### -S -emit-llvm -G 0 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECK32 %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### -c -G 0 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECK32 %s
// RUN: touch %t.s %t.o
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### -c %t.s -G 0 2>&1 | FileCheck --check-prefix=CHECK32 %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### -o dummy.so %t.o -G 0 2>&1 | FileCheck --check-prefix=CHECK32 %s

// RUN: %clang --target=powerpc64-ibm-aix-xcoff -### -E -G 0 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECK64 %s
// RUN: %clang --target=powerpc64-ibm-aix-xcoff -### -S -emit-llvm -G 0 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECK64 %s
// RUN: %clang --target=powerpc64-ibm-aix-xcoff -### -c -G 0 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECK64 %s
// RUN: %clang --target=powerpc64-ibm-aix-xcoff -### -c %t.s -G 0 2>&1 | FileCheck --check-prefix=CHECK64 %s
// RUN: %clang --target=powerpc64-ibm-aix-xcoff -### -o dummy.so %t.o -G 0 2>&1 | FileCheck --check-prefix=CHECK64 %s

// CHECK32: error: unsupported option '-G' for target 'powerpc-ibm-aix-xcoff'
// CHECK64: error: unsupported option '-G' for target 'powerpc64-ibm-aix-xcoff'
