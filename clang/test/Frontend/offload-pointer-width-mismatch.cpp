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

// Every type that disagrees with the device data layout is noted.

// AMDGCN: error: device target 'amdgcn-amd-amdhsa' is not compatible with host target 'i386-unknown-linux-gnu'
// AMDGCN-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)
// AMDGCN-NEXT: note: alignment of type 'void *' for the host target (4 bytes) does not match the alignment for the device target (8 bytes)
// AMDGCN-NEXT: note: size of type 'size_t' for the host target (4 bytes) does not match the size for the device target (8 bytes)
// AMDGCN-NEXT: note: size of type 'ptrdiff_t' for the host target (4 bytes) does not match the size for the device target (8 bytes)
// AMDGCN-NEXT: note: size of type 'intptr_t' for the host target (4 bytes) does not match the size for the device target (8 bytes)

// R600: error: device target 'r600-unknown-unknown' is not compatible with host target 'x86_64-unknown-linux-gnu'
// R600-NEXT: note: size of type 'void *' for the host target (8 bytes) does not match the size for the device target (4 bytes)
// NVPTX: error: device target 'nvptx64-nvidia-cuda' is not compatible with host target 'i386-unknown-linux-gnu'
// NVPTX-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)
// SPIR: error: device target 'spir-unknown-unknown' is not compatible with host target 'x86_64-unknown-linux-gnu'
// SPIR-NEXT: note: size of type 'void *' for the host target (8 bytes) does not match the size for the device target (4 bytes)
// SPIRV: error: device target 'spirv64-unknown-unknown' is not compatible with host target 'i386-unknown-linux-gnu'
// SPIRV-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)

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

// SPIRV-NVPTX: error: device target 'spirv64-unknown-unknown' is not compatible with host target 'nvptx-nvidia-cuda'
// SPIRV-NVPTX-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)
// AMDGCN-R600: error: device target 'amdgcn-amd-amdhsa' is not compatible with host target 'r600-unknown-unknown'
// AMDGCN-R600-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)
// AMDHSA-SPIR: error: device target 'spirv64-amd-amdhsa' is not compatible with host target 'spir-unknown-unknown'
// AMDHSA-SPIR-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)

// The host pointer width is taken from the host target rather than from its
// triple, so an ABI that narrows pointers is caught too.

// RUN: not %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnux32 \
// RUN:   -fsycl-is-device -fsyntax-only %s 2>&1 | FileCheck --check-prefix=SPIR64-X32 %s
// RUN: not %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnux32 \
// RUN:   -fcuda-is-device -fsyntax-only -x cuda %s 2>&1 | FileCheck --check-prefix=NVPTX64-X32 %s
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnux32 \
// RUN:   -fcuda-is-device -fsyntax-only -x hip %s 2>&1 | FileCheck --check-prefix=AMDGCN-X32 %s

// SPIR64-X32: error: device target 'spir64-unknown-unknown' is not compatible with host target 'x86_64-unknown-linux-gnux32'
// SPIR64-X32-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)
// NVPTX64-X32: error: device target 'nvptx64-nvidia-cuda' is not compatible with host target 'x86_64-unknown-linux-gnux32'
// NVPTX64-X32-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)
// AMDGCN-X32: error: device target 'amdgcn-amd-amdhsa' is not compatible with host target 'x86_64-unknown-linux-gnux32'
// AMDGCN-X32-NEXT: note: size of type 'void *' for the host target (4 bytes) does not match the size for the device target (8 bytes)

// NVPTX and AMDGPU take the pointer alignment from the host too, so a host that
// under aligns pointers is a mismatch even where the widths agree.

// RUN: not %clang_cc1 -triple nvptx-nvidia-cuda -aux-triple m68k-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x cuda %s 2>&1 | FileCheck --check-prefix=NVPTX-M68K %s
// RUN: not %clang_cc1 -triple r600-unknown-unknown -aux-triple m68k-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fsyntax-only -x hip %s 2>&1 | FileCheck --check-prefix=R600-M68K %s

// The alignment is then the only type detail noted.

// NVPTX-M68K: error: device target 'nvptx-nvidia-cuda' is not compatible with host target 'm68k-unknown-linux-gnu'
// NVPTX-M68K-NEXT: note: alignment of type 'void *' for the host target (2 bytes) does not match the alignment for the device target (4 bytes)
// NVPTX-M68K-NOT: note:
// R600-M68K: error: device target 'r600-unknown-unknown' is not compatible with host target 'm68k-unknown-linux-gnu'
// R600-M68K-NEXT: note: alignment of type 'void *' for the host target (2 bytes) does not match the alignment for the device target (4 bytes)
// R600-M68K-NOT: note:

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
