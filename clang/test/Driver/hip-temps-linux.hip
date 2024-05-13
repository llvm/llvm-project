// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: system-linux

// Check no temporary files or directores are left after compilation.
// RUN: rm -rf %t/mytmp
// RUN: mkdir -p %t/mytmp
// RUN: env TMPDIR="%t/mytmp" %clang --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --rocm-path=%S/Inputs/rocm -nostdinc -nostdlib -c \
// RUN:   --offload-arch=gfx1030 -emit-llvm -v %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK %s
// RUN: ls %t/mytmp >%t/mytmp.txt 2>&1
// RUN: touch %t/empty.txt
// RUN: diff %t/mytmp.txt %t/empty.txt

// CHECK: -o {{.*}}/mytmp/hip-temps-linux-gfx1030-{{.*}}.bc

int main() {}
