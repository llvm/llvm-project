// RUN: %clang -### -c --target=x86_64 -mfpmath=sse %s 2>&1 | FileCheck %s
// CHECK: "-mfpmath" "sse"

// RUN: %clang -### -c --target=x86_64 -mfpmath=sse -x assembler %s 2>&1 | FileCheck %s --check-prefix=WARN
// WARN: warning: argument unused during compilation: '-mfpmath=sse'
