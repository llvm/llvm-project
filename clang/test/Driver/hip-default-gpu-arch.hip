// RUN: %clang -### -nogpulib -nogpuinc -c %s 2>&1 | FileCheck %s

// CHECK: {{.*}}clang{{.*}}"-triple" "{{amdgcn|spirv64}}-amd-amdhsa"{{.*}} "-target-cpu" "{{amdgcnspirv|gfx.*}}"
