// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Check no temporary files or directories are left after compilation.

// RUN: rm -rf %t%{fs-sep}mytmp
// RUN: mkdir -p %t%{fs-sep}mytmp
// RUN: env TMPDIR="%t/mytmp" TEMP="%t/mytmp" TMP="%t/mytmp" %clang --target=x86_64-linux-gnu \
// RUN:   -nogpulib -nogpuinc -nostdinc -nostdlib \
// RUN:   --rocm-path=%S/Inputs/rocm -c \
// RUN:   --offload-arch=gfx1030 -emit-llvm -v %s 2>&1 | \
// RUN:   FileCheck -DTMPDIR=%t%{fs-sep}mytmp%{fs-sep} %s
// RUN: ls %t/mytmp >%t/mytmp.txt 2>&1
// RUN: touch %t/empty.txt
// RUN: diff %t/mytmp.txt %t/empty.txt

// CHECK: -o {{.*}}[[TMPDIR]]hip-temps-linux-gfx1030-{{.*}}.bc

int main() {}
