// REQUIRES: amdgpu-registered-target
// REQUIRES: lld

// RUN: %clang --target=amdgcn-amd-amdhsa -emit-llvm -c -nogpulib -DVAR=x %s -o %t.x.bc
// RUN: %clang --target=amdgcn-amd-amdhsa -emit-llvm -c -nogpulib -DVAR=y %s -o %t.y.bc
// RUN: llvm-offload-binary -o %t.x.bundle.bc \
// RUN:   --image=file=%t.x.bc,triple=amdgcn-amd-amdhsa,arch=gfx906,kind=hip \
// RUN:   --image=file=%t.x.bc,triple=amdgcn-amd-amdhsa,arch=gfx942,kind=hip
// RUN: llvm-offload-binary -o %t.y.bundle.bc \
// RUN:   --image=file=%t.y.bc,triple=amdgcn-amd-amdhsa,arch=gfx906,kind=hip \
// RUN:   --image=file=%t.y.bc,triple=amdgcn-amd-amdhsa,arch=gfx942,kind=hip

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --hip-link --cuda-device-only \
// RUN:   --offload-arch=gfx906 --offload-arch=gfx942 \
// RUN:   %t.x.bundle.bc %t.y.bundle.bc -o %t.hipfb 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DRIVER
// DRIVER: "{{.*}}clang-linker-wrapper"
// DRIVER-SAME: "--should-extract=gfx906"
// DRIVER-SAME: "--should-extract=gfx942"
// DRIVER-SAME: "--emit-fatbin-only" "-o" "{{.*}}.hipfb"
// DRIVER-SAME: "{{.*}}.x.bundle.bc" "{{.*}}.y.bundle.bc"

// RUN: %clang -### -v --target=x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --hip-link --cuda-device-only \
// RUN:   --offload-arch=gfx906 %t.x.bundle.bc -o %t.hipfb 2>&1 \
// RUN:   | FileCheck %s --check-prefix=VERBOSE
// VERBOSE: "{{.*}}clang-linker-wrapper"
// VERBOSE-SAME: "--device-compiler=amdgcn-amd-amdhsa=-v"
// VERBOSE-SAME: "--wrapper-verbose"
// VERBOSE-SAME: "--emit-fatbin-only"

// RUN: %clang --target=x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --hip-link --cuda-device-only \
// RUN:   --offload-arch=gfx906 --offload-arch=gfx942 \
// RUN:   %t.x.bundle.bc %t.y.bundle.bc -o %t.hipfb
// RUN: clang-offload-bundler -type=o -list -input=%t.hipfb \
// RUN:   | FileCheck %s --check-prefix=ARCH
// ARCH-DAG: hip-amdgcn-amd-amdhsa--gfx906
// ARCH-DAG: hip-amdgcn-amd-amdhsa--gfx942

// RUN: %clang -### -c --target=x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --hip-link --cuda-device-only \
// RUN:   --offload-arch=gfx906 %t.x.bundle.bc -o %t.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=COMPILE
// COMPILE: "-cc1"
// COMPILE-NOT: "{{.*}}clang-linker-wrapper"

// RUN: %clang -### -emit-llvm --target=x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --hip-link --cuda-device-only \
// RUN:   --offload-arch=gfx906 %t.x.bundle.bc -o %t.linked.bc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-FATBIN --allow-empty
// RUN: %clang -### --no-gpu-bundle-output \
// RUN:   --target=x86_64-unknown-linux-gnu -fgpu-rdc --hip-link \
// RUN:   --cuda-device-only --offload-arch=gfx906 %t.x.bundle.bc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-FATBIN --allow-empty
// NO-FATBIN-NOT: "--emit-fatbin-only"

__attribute__((visibility("protected"), used)) int VAR;
