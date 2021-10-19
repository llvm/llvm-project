// RUN: %clang -### -S -fno-visibility-inlines-hidden -fvisibility-inlines-hidden %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-1 %s
// CHECK-1: "-fvisibility-inlines-hidden"

// RUN: %clang -### -S -fvisibility-inlines-hidden -fno-visibility-inlines-hidden %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-2 %s
// CHECK-2-NOT: "-fvisibility-inlines-hidden"
