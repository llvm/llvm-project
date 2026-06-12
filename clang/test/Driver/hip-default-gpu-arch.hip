// RUN: %clang -### -nogpulib -nogpuinc -c %s 2>&1 | FileCheck %s

// CHECK: {{.*}}clang{{.*}}"-triple" "amdgpu9.06
