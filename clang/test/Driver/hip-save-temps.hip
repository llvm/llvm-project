// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// -fno-gpu-rdc without -o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   -x hip --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NORDC,NOUT %s

// -fno-gpu-rdc with -o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   -o executable -x hip --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NORDC,WOUT %s

// -fgpu-rdc without -o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   -fgpu-rdc -x hip --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,RDC,RDC-NOUT,NOUT %s

// -fgpu-rdc with -o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   -o executable -fgpu-rdc -x hip --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,RDC,RDC-WOUT,WOUT %s

// CHECK: {{.*}}clang{{.*}}"-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.cui"
// CHECK: {{.*}}llvm-link{{.*}}"-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900-linked.bc"
// CHECK: {{.*}}opt{{.*}}"-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900-optimized.bc"
// CHECK: {{.*}}llc{{.*}}"-filetype=asm"{{.*}}"-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.s"
// CHECK: {{.*}}llc{{.*}}"-filetype=obj"{{.*}}"-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.o"
// NORDC: {{.*}}lld{{.*}}"-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.out"
// RDC: {{.*}}lld{{.*}}"-o" "a.out-hip-amdgcn-amd-amdhsa-gfx900"
// NORDC: {{.*}}clang-offload-bundler{{.*}}"-outputs=hip-save-temps.hip-hip-amdgcn-amd-amdhsa.hipfb"
// CHECK: {{.*}}clang{{.*}}"-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.cui"
// CHECK: {{.*}}clang{{.*}}"-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.bc"
// CHECK: {{.*}}clang{{.*}}"-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.s"
// CHECK: {{.*}}clang{{.*}}"-o" "hip-save-temps{{.*}}.o"
// RDC-NOUT: {{.*}}clang-offload-bundler{{.*}}"-outputs=a.out.hipfb"
// RDC-WOUT: {{.*}}clang-offload-bundler{{.*}}"-outputs=executable.hipfb"
// NOUT: {{.*}}ld{{.*}}"-o" "a.out"
// WOUT: {{.*}}ld{{.*}}"-o" "executable"

