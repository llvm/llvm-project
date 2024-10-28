// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Check no temporary files or directories are left after compilation.

// RUN: rm -rf %t%{fs-sep}mytmp
// RUN: mkdir -p %t%{fs-sep}mytmp
// RUN: env TMPDIR="%t%{fs-sep}mytmp" TEMP="%t%{fs-sep}mytmp" TMP="%t%{fs-sep}mytmp" %clang --target=x86_64-linux-gnu \
// RUN:   -nogpulib -nogpuinc -nostdinc -nostdlib \
// RUN:   --rocm-path=%S/Inputs/rocm -c \
// RUN:   --offload-arch=gfx1030 -emit-llvm -v %s 2>&1 | \
// RUN:   FileCheck -DTMPDIR=%t%{fs-sep}mytmp%{fs-sep} %s
// RUN: ls %t%{fs-sep}mytmp >%t%{fs-sep}mytmp.txt 2>&1
// RUN: touch %t%{fs-sep}empty.txt
// RUN: diff %t%{fs-sep}mytmp.txt %t%{fs-sep}empty.txt

// CHECK: -o {{.*}}[[TMPDIR]]hip-temps-linux-gfx1030-{{.*}}.bc

int main() {}
