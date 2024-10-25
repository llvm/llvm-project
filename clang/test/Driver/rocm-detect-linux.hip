// REQUIRES: system-linux

// Test to ensure that on Windows, we do not include linux sesrch paths
// RUN: %clang -### -nogpulib -nogpuinc \
// RUN:  --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck %s --dump-input-context 100

// CHECK: ROCm installation search path: {{/usr/local}}
// CHECK: ROCm installation search path: {{/usr}}
