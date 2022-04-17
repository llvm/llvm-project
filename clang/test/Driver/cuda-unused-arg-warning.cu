// Tests that we trigger unused-arg warnings on CUDA flags appropriately.

// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// --cuda-host-only and --cuda-compile-host-device should never trigger an
// unused arg warning.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only -c %s 2>&1 | \
// RUN:    FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only -x c -c %s 2>&1 | \
// RUN:    FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu --cuda-compile-host-device -c %s 2>&1 | \
// RUN:    FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu --cuda-compile-host-device -x c -c %s 2>&1 | \
// RUN:    FileCheck %s

// --cuda-device-only should warn during non-CUDA compilation.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only -x c -c %s 2>&1 | \
// RUN:    FileCheck -check-prefix UNUSED-WARNING %s

// --cuda-device-only should not produce warning compiling CUDA files
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only -c %s 2>&1 | \
// RUN:    FileCheck -check-prefix NO-UNUSED-WARNING %s

// CHECK-NOT: warning: argument unused during compilation: '--cuda-host-only'
// CHECK-NOT: warning: argument unused during compilation: '--cuda-compile-host-device'
// UNUSED-WARNING: warning: argument unused during compilation: '--cuda-device-only'
// NO-UNUSED-WARNING-NOT: warning: argument unused during compilation: '--cuda-device-only'
