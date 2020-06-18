// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// -fno-gpu-rdc without -o with -c
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --cuda-gpu-arch=gfx900 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NORDC %s

// -fno-gpu-rdc without -o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NORDC,NOUT %s

// -fno-gpu-rdc with -o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   -o executable --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NORDC,WOUT %s

// -fgpu-rdc without -o with -c
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   -fgpu-rdc --cuda-gpu-arch=gfx900 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,RDC,RDCC %s

// -fgpu-rdc without -o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   -fgpu-rdc --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,RDC,RDCL,RDC-NOUT,NOUT %s

// -fgpu-rdc with -o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   -o executable -fgpu-rdc --cuda-gpu-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,RDC,RDCL,RDC-WOUT,WOUT %s

// CHECK: {{".*clang.*"}} "-cc1" {{.*}} "-E" {{.*}} [[CPU:"-target-cpu" "gfx900"]] {{.*}}  "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.cui"
// NORDC: {{".*clang.*"}} "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} [[CPU]] {{.*}} "-disable-llvm-passes" {{.*}} "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.bc"
// RDC: {{".*clang.*"}} "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} [[CPU]] {{.*}} "-disable-llvm-passes" {{.*}} "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.tmp.bc"

// NORDC: {{".*clang.*"}} "-cc1" {{.*}} "-S"
// NORDC-SAME: "-mllvm" "-amdgpu-internalize-symbols"
// NORDC-SAME: [[CPU]]
// NORDC-SAME: "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.s"

// RDC: {{".*clang.*"}} "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} [[CPU]] {{.*}} "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.bc"
// NORDC: {{".*clang.*"}} "-cc1as" {{.*}} "-filetype" "obj" {{.*}} [[CPU]] {{.*}} "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.o"
// CHECK-NOT: llvm-link
// CHECK-NOT: opt
// CHECK-NOT: llc
// NORDC: {{.*lld.*}}"-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.out"

// RDCL: "{{.*lld.*}}" {{.*}} "-mllvm" "-amdgpu-internalize-symbols"
// RDCL-SAME: "-o" "a.out-hip-amdgcn-amd-amdhsa-gfx900"

// NORDC: "{{.*clang-offload-bundler.*}}" {{.*}} "-outputs=hip-save-temps.hip-hip-amdgcn-amd-amdhsa.hipfb"
// CHECK: "{{.*clang.*}}" "-cc1" {{.*}} "-E" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.cui"
// NORDC: "{{.*clang.*}}" "-cc1" {{.*}} "-emit-llvm-bc" {{.*}}  "-fcuda-include-gpubinary" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.bc"
// RDC: "{{.*clang.*}}" "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.bc"
// CHECK: "{{.*clang.*}}" "-cc1" {{.*}} "-S" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.s"
// CHECK: "{{.*clang.*}}" "-cc1as" {{.*}} "-o" "hip-save-temps{{.*}}.o"
// RDCC: "{{.*clang-offload-bundler.*}}" {{.*}} "-outputs=hip-save-temps.o"
// RDC-NOUT: "{{.*clang-offload-bundler.*}}" {{.*}} "-outputs=a.out.hipfb"
// RDC-WOUT: "{{.*clang-offload-bundler.*}}" {{.*}} "-outputs=executable.hipfb"
// NOUT: "{{.*ld.*}}" {{.*}} "-o" "a.out"
// WOUT: "{{.*ld.*}}" {{.*}} "-o" "executable"