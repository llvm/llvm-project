// RUN: %clang -target powerpc32-ibm-aix-xcoff -### -S -emit-llvm -G 0 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECK32 %s
// RUN: %clang -target powerpc64-ibm-aix-xcoff -### -S -emit-llvm -G 0 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECK64 %s

// CHECK32: error: unsupported option '-G' for target 'powerpc32-ibm-aix-xcoff'
// CHECK64: error: unsupported option '-G' for target 'powerpc64-ibm-aix-xcoff'
