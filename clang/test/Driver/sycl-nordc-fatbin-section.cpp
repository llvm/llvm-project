/// Verify which section carries the SYCL device image in the host object.
// REQUIRES: spirv-registered-target, x86-registered-target

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl-rdc \
// RUN:   -c %s -o %t.nordc.o
// RUN: llvm-readelf -S %t.nordc.o \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'
// NORDC: .sycl_fatbin

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fsycl-rdc \
// RUN:   -c %s -o %t.rdc.o
// RUN: llvm-readelf -S %t.rdc.o \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.sycl_fatbin'
// RDC: .llvm.offloading

/// The choice does not depend on the object format.
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-rdc \
// RUN:   -c %s -o %t.nordc.obj
// RUN: llvm-readobj --sections %t.nordc.obj \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'

// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl -fsycl-rdc \
// RUN:   -c %s -o %t.rdc.obj
// RUN: llvm-readobj --sections %t.rdc.obj \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.sycl_fatbin'

void f() {}
