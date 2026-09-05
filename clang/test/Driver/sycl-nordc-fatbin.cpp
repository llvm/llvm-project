// REQUIRES: spirv-registered-target, x86-registered-target

/// Verify which section carries the SYCL device binary in the host object.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc \
// RUN:   -c %s -o %t.nordc.o
// RUN: llvm-readelf -S %t.nordc.o \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fgpu-rdc \
// RUN:   -c %s -o %t.rdc.o
// RUN: llvm-readelf -S %t.rdc.o \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.sycl_fatbin'

/// The choice does not depend on the object format.
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl -fno-gpu-rdc \
// RUN:   -c %s -o %t.nordc.obj
// RUN: llvm-readobj --sections %t.nordc.obj \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'

// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl -fgpu-rdc \
// RUN:   -c %s -o %t.rdc.obj
// RUN: llvm-readobj --sections %t.rdc.obj \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.sycl_fatbin'

// NORDC: .sycl_fatbin
// RDC: .llvm.offloading

/// The section holds an offload binary (magic 0x10FF10AD) whose image is a
/// finalized SPIR-V module (magic 0x07230203), both shown little endian by the
/// hex dump.
// RUN: llvm-readelf --hex-dump=.sycl_fatbin %t.nordc.o \
// RUN:   | FileCheck -check-prefix=BINARY %s

// BINARY: 10ff10ad
// BINARY: 03022307

void f() {}
