// REQUIRES: amdgpu-registered-target, nvptx-registered-target
// REQUIRES: spirv-registered-target, x86-registered-target

// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple i386-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x hip %s 2>&1 | FileCheck --check-prefix=AMDGCN %s
// RUN: not %clang_cc1 -triple r600-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x hip %s 2>&1 | FileCheck --check-prefix=R600 %s
// RUN: not %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple i386-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x cuda %s 2>&1 | FileCheck --check-prefix=NVPTX %s
// RUN: not %clang_cc1 -triple spir-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -fsyntax-only %s 2>&1 | FileCheck --check-prefix=SPIR %s
// RUN: not %clang_cc1 -triple spirv64-unknown-unknown -aux-triple i386-unknown-linux-gnu \
// RUN:   -fsycl-is-device -fsyntax-only %s 2>&1 | FileCheck --check-prefix=SPIRV %s

// AMDGCN: error: device target 'amdgcn-amd-amdhsa' takes a pointer width of 32 bits from host target 'i386-unknown-linux-gnu', but requires 64 bits
// R600: error: device target 'r600-unknown-unknown' takes a pointer width of 64 bits from host target 'x86_64-unknown-linux-gnu', but requires 32 bits
// NVPTX: error: device target 'nvptx64-nvidia-cuda' takes a pointer width of 32 bits from host target 'i386-unknown-linux-gnu', but requires 64 bits
// SPIR: error: device target 'spir-unknown-unknown' takes a pointer width of 64 bits from host target 'x86_64-unknown-linux-gnu', but requires 32 bits
// SPIRV: error: device target 'spirv64-unknown-unknown' takes a pointer width of 32 bits from host target 'i386-unknown-linux-gnu', but requires 64 bits

// OpenMP device compilation adapts at both stages too.

// RUN: not %clang_cc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple i386-unknown-linux-gnu -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck --check-prefix=AMDGCN %s
// RUN: not %clang_cc1 -fopenmp -fopenmp-is-target-device -triple nvptx64-nvidia-cuda \
// RUN:   -aux-triple i386-unknown-linux-gnu -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NVPTX %s

// A host target that is itself a device target is no exception.

// RUN: not %clang_cc1 -triple spirv64-unknown-unknown -aux-triple nvptx-nvidia-cuda \
// RUN:   -fsycl-is-device -fsyntax-only %s 2>&1 | FileCheck --check-prefix=SPIRV-NVPTX %s
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple r600-unknown-unknown \
// RUN:   -fcuda-is-device -fsyntax-only -x hip %s 2>&1 | FileCheck --check-prefix=AMDGCN-R600 %s
// RUN: not %clang_cc1 -triple spirv64-amd-amdhsa -aux-triple spir-unknown-unknown \
// RUN:   -fsycl-is-device -fsyntax-only %s 2>&1 | FileCheck --check-prefix=AMDHSA-SPIR %s

// SPIRV-NVPTX: error: device target 'spirv64-unknown-unknown' takes a pointer width of 32 bits from host target 'nvptx-nvidia-cuda', but requires 64 bits
// AMDGCN-R600: error: device target 'amdgcn-amd-amdhsa' takes a pointer width of 32 bits from host target 'r600-unknown-unknown', but requires 64 bits
// AMDHSA-SPIR: error: device target 'spirv64-amd-amdhsa' takes a pointer width of 32 bits from host target 'spir-unknown-unknown', but requires 64 bits

// The host pointer width is taken from the host target rather than from its
// triple, so an ABI that narrows pointers is caught too.

// RUN: not %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnux32 \
// RUN:   -fsycl-is-device -fsyntax-only %s 2>&1 | FileCheck --check-prefix=SPIR64-X32 %s
// RUN: not %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnux32 \
// RUN:   -fcuda-is-device -fsyntax-only -x cuda %s 2>&1 | FileCheck --check-prefix=NVPTX64-X32 %s
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnux32 \
// RUN:   -fcuda-is-device -fsyntax-only -x hip %s 2>&1 | FileCheck --check-prefix=AMDGCN-X32 %s

// SPIR64-X32: error: device target 'spir64-unknown-unknown' takes a pointer width of 32 bits from host target 'x86_64-unknown-linux-gnux32', but requires 64 bits
// NVPTX64-X32: error: device target 'nvptx64-nvidia-cuda' takes a pointer width of 32 bits from host target 'x86_64-unknown-linux-gnux32', but requires 64 bits
// AMDGCN-X32: error: device target 'amdgcn-amd-amdhsa' takes a pointer width of 32 bits from host target 'x86_64-unknown-linux-gnux32', but requires 64 bits

// NVPTX and AMDGPU take the pointer alignment from the host too, so a host that
// under aligns pointers is a mismatch even where the widths agree.

// RUN: not %clang_cc1 -triple nvptx-nvidia-cuda -aux-triple m68k-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x cuda %s 2>&1 | FileCheck --check-prefix=NVPTX-M68K %s
// RUN: not %clang_cc1 -triple r600-unknown-unknown -aux-triple m68k-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x hip %s 2>&1 | FileCheck --check-prefix=R600-M68K %s

// NVPTX-M68K: error: device target 'nvptx-nvidia-cuda' takes a pointer alignment of 16 bits from host target 'm68k-unknown-linux-gnu', but requires 32 bits
// R600-M68K: error: device target 'r600-unknown-unknown' takes a pointer alignment of 16 bits from host target 'm68k-unknown-linux-gnu', but requires 32 bits

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x hip %s
// RUN: %clang_cc1 -triple r600-unknown-unknown -aux-triple i386-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x hip %s
// RUN: %clang_cc1 -triple nvptx-nvidia-cuda -aux-triple i386-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x cuda %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -fsyntax-only %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -fsyntax-only %s
// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsyntax-only %s

// SPIR and SPIR-V align pointers to their own width rather than to the host
// alignment, so an under aligning host is not a mismatch there.

// RUN: %clang_cc1 -triple spir-unknown-unknown -aux-triple m68k-unknown-linux-gnu \
// RUN:   -fsycl-is-device -fsyntax-only %s

// SPIR, SPIR-V, and NVPTX keep their own pointer related types when the host is
// of their own family, so a differing width is not a mismatch there.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple spir-unknown-unknown \
// RUN:   -fsycl-is-device -fsyntax-only %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple nvptx-nvidia-cuda \
// RUN:   -fcuda-is-device -fsyntax-only -x cuda %s
