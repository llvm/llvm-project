// UNSUPPORTED: system-windows
//
// Copying clang to a new location and running it will not work unless it is
// statically linked. Dynamically linked builds typically use relative rpaths,
// which this will break.
// REQUIRES: static-libs
//
// Test detecting ROCm installation from clang installed in /opt/rocm/lib/llvm/bin.
// This exercises the RocmInstallationDetector logic that backs up from
// /opt/rocm/lib/llvm/bin to the ROCm root, while avoiding dependence on any
// real /opt/rocm installation by using a synthetic sysroot under %t.
//
// RUN: rm -rf %t/*
// Set up a fake ROCm installation rooted at /opt/rocm in the synthetic sysroot
// so that the device libraries and HIP runtime can be found.
// RUN: mkdir -p %t/sysroot/opt/rocm
// RUN: cp -r %S/Inputs/rocm/* %t/sysroot/opt/rocm
// Copy the built clang binary so that its InstalledDir and the real path used
// by the driver both reside under /opt/rocm/lib/llvm/bin, which is what the
// detection logic inspects.
// RUN: mkdir -p %t/sysroot/opt/rocm/lib/llvm/bin
// RUN: cp %clang %t/sysroot/opt/rocm/lib/llvm/bin/clang
// RUN: %t/sysroot/opt/rocm/lib/llvm/bin/clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx1010 --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-LIB-LLVM %s
//
// ROCM-LIB-LLVM: ROCm installation search path: {{.*sysroot/opt/rocm$}}
// ROCM-LIB-LLVM-NOT: ROCm installation search path: {{.*sysroot/opt/rocm/lib$}}
