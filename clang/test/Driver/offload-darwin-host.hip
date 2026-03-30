// Make sure the darwin host toolchain does not assert with offload
// languages. The darwin toolchain has a platform initialization hack which was
// broken by offload toolchains trying to perform extra host argument handling.

// RUN: %clang -### --target=arm64-apple-darwin24.6.0 -mmacos-version-min=10.9 --offload-arch=gfx900 -nogpulib -nogpuinc %s 2>&1 | FileCheck %s

// CHECK: "-cc1" "-triple" "amdgpu9.00-amd-amdhsa" "-aux-triple" "arm64-apple-darwin24.6.0"
// CHECK: "-cc1" "-triple" "arm64-apple-macosx10.9.0" "-aux-triple" "amdgpu-amd-amdhsa"
