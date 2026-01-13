// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// UNSUPPORTED: system-windows

// Check no temporary files or directores are left after compilation.
// RUN: rm -rf %t/mytmp
// RUN: mkdir -p %t/mytmp
// RUN: env TMP="%t/mytmp" TMPDIR="%t/mytmp" %clang --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --rocm-path=%S/Inputs/rocm -nostdinc -nostdlib -c \
// RUN:   --offload-arch=gfx1030 -emit-llvm -v %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK -DOUTPUT_PATH="%t%{fs-sep}mytmp%{fs-sep}" %s
// RUN: ls %t/mytmp >%t/mytmp.txt 2>&1
// RUN: touch %t/empty.txt
// RUN: diff %t/mytmp.txt %t/empty.txt

// CHECK: -o {{"?}}[[OUTPUT_PATH]]hip-temps-linux-gfx1030-{{.*}}.bc{{"?}}

int main() {}
