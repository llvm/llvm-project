// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// -fno-gpu-rdc without -o with -c
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --no-offload-new-driver -nogpuinc --offload-arch=gfx900 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NORDC %s

// -fno-gpu-rdc without -o
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --no-offload-new-driver -nogpuinc --offload-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NORDC,NOUT %s

// -fno-gpu-rdc with -o
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --no-offload-new-driver -nogpuinc -o executable --offload-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NORDC,WOUT %s

// -fgpu-rdc without -o with -c
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --no-offload-new-driver -nogpuinc -fgpu-rdc --offload-arch=gfx900 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,RDC,RDCC %s

// -fgpu-rdc without -o
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --no-offload-new-driver -nogpuinc -fgpu-rdc --offload-arch=gfx900 %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,RDC,RDCL,NOUT %s

// -fgpu-rdc with -o
// UN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// UN:   --offload-new-driver -nogpuinc -o executable -fgpu-rdc --offload-arch=gfx900 %s 2>&1 | \
// UN:   FileCheck -check-prefixes=CHECK,RDC,RDCL,WOUT %s

// -fgpu-rdc host object path
// RDCL: "{{.*clang.*}}" "-cc1" {{.*}} "-E" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.hipi"
// RDCL: "{{.*clang.*}}" "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.bc"
// RDCL: "{{.*clang.*}}" "-cc1" {{.*}} "-S" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.s"
// RDCL: "{{.*clang.*}}" "-cc1as" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.o"

// device object paths
// CHECK: {{".*clang.*"}} "-cc1" {{.*}} "-E" {{.*}} [[CPU:"-target-cpu" "gfx900"]] {{.*}}  "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.hipi"
// NORDC: {{".*clang.*"}} "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} [[CPU]] {{.*}} "-disable-llvm-passes" {{.*}} "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.bc"
// RDC: {{".*clang.*"}} "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} [[CPU]] {{.*}} "-disable-llvm-passes" {{.*}} "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.tmp.bc"

// NORDC: {{".*clang.*"}} "-cc1" {{.*}} "-S"
// NORDC-SAME: "-mllvm" "-amdgpu-internalize-symbols"
// NORDC-SAME: [[CPU]]
// NORDC-SAME: "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.s"

// RDC: {{".*clang.*"}} "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} [[CPU]] {{.*}} "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.bc"
// NORDC: {{".*clang.*"}} "-cc1as" {{.*}} "-filetype" "obj" {{.*}} [[CPU]] {{.*}} "-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.o"
// CHECK-NOT: "{{.*}}llvm-link"
// CHECK-NOT: "{{.*}}opt"
// CHECK-NOT: "{{.*}}llc"
// NORDC: {{.*lld.*}}"-o" "hip-save-temps-hip-amdgcn-amd-amdhsa-gfx900.out"
// RDCL: "{{.*lld.*}}" {{.*}} "-plugin-opt=-amdgpu-internalize-symbols" {{.*}}"-save-temps"
// RDCL-SAME: "-o" "a.out-hip-amdgcn-amd-amdhsa-gfx900"
// RDCC: "{{.*clang.*}}" "-cc1" {{.*}} "-E" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.hipi"
// RDCC: "{{.*clang.*}}" "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.bc"
// RDCC: "{{.*clang.*}}" "-cc1" {{.*}} "-S" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.s"
// RDCC: "{{.*clang.*}}" "-cc1as" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.o"
// RDCC: "{{.*clang-offload-bundler.*}}" {{.*}} "-output=hip-save-temps.o"
// RDCL: "{{.*clang-offload-bundler.*}}" {{.*}} "-output=hip-save-temps-hip-amdgcn-amd-amdhsa.hipfb"
// RDCL: {{.*}}llvm-mc{{.*}}"-o" "hip-save-temps-hip-amdgcn-amd-amdhsa.o" "hip-save-temps-hip-amdgcn-amd-amdhsa.mcin" "--filetype=obj"

// -fno-gpu-rdc host object path
// NORDC: "{{.*clang.*}}" "-cc1" {{.*}} "-E" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.hipi"
// NORDC: "{{.*clang.*}}" "-cc1" {{.*}} "-emit-llvm-bc" {{.*}}  "-fcuda-include-gpubinary" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.bc"
// NORDC: "{{.*clang.*}}" "-cc1" {{.*}} "-S" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.s"
// NORDC: "{{.*clang.*}}" "-cc1as" {{.*}} "-o" "hip-save-temps{{.*}}.o"

// output to default a.out or -o specified file name
// NOUT: {{.*}}ld{{.*}}"-o" "a.out"
// WOUT: {{.*}}ld{{.*}}"-o" "executable"
