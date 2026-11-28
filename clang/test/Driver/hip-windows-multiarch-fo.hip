// REQUIRES: system-windows
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Each offload-arch device compile job must get its own unique output path
// under /Fo, the same way -o already does via GetNamedOutputPath's temp-file
// gate. Otherwise every arch's device object collides on the single literal
// /Fo path and silently overwrites the others before packaging.

// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -x hip \
// RUN:   --offload-arch=gfx900 --offload-arch=gfx90a:xnack+ \
// RUN:   -nogpuinc -nogpulib -Foout.obj -- %s 2>&1 | FileCheck %s

// CHECK: "-target-cpu" "gfx900"{{.*}}"-o" "{{.*}}gfx900-{{[^"]*}}"
// CHECK-NOT: "-o" "out.obj"
// CHECK: "-target-cpu" "gfx90a" "-mxnack"{{.*}}"-o" "{{.*}}gfx90a@xnack+-{{[^"]*}}"
// CHECK-NOT: "-o" "out.obj"
// CHECK: "-o" "out.obj"

void main() {}
