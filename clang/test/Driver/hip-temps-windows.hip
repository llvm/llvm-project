// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: system-windows

// Check no temporary files or directores are left after compilation.
// RUN: rm -rf %t/mytmp
// RUN: mkdir -p %t/mytmp
// RUN: env TMP="%t/mytmp" %clang --target=x86_64-pc-windows-msvc -nogpulib -nogpuinc \
// RUN:   --rocm-path=%S/Inputs/rocm -nostdinc -nostdlib -c \
// RUN:   --offload-arch=gfx1030 -emit-llvm -v %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK %s
// RUN: ls %t/mytmp >%t/mytmp.txt 2>&1
// RUN: touch %t/empty.txt
// RUN: diff %t/mytmp.txt %t/empty.txt

// CHECK: -o "{{.*}}mytmp{{/|\\\\}}hip-temps-windows-gfx1030-{{.*}}.bc"

int main() {}
