// RUN: %clang -### -nogpulib -nogpuinc -c %s 2>&1 | FileCheck %s

// CHECK: {{.*}}clang{{.*}}"-target-cpu" "gfx906"
