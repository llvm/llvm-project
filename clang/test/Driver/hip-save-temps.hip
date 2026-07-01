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
// CHECK: {{".*clang.*"}} "-cc1" "-triple" "amdgpu9.00-amd-amdhsa" {{.*}} "-E" {{.*}} "-o" "hip-save-temps-hip-amdgpu9.00-amd-amdhsa-gfx900.hipi"
// NORDC: {{".*clang.*"}} "-cc1" "-triple" "amdgpu9.00-amd-amdhsa" {{.*}} "-emit-llvm-bc" {{.*}} "-disable-llvm-passes" {{.*}} "-o" "hip-save-temps-hip-amdgpu9.00-amd-amdhsa-gfx900.bc"
// RDC: {{".*clang.*"}} "-cc1" "-triple" "amdgpu9.00-amd-amdhsa" {{.*}} "-emit-llvm-bc" {{.*}} "-disable-llvm-passes" {{.*}} "-o" "hip-save-temps-hip-amdgpu9.00-amd-amdhsa-gfx900.tmp.bc"

// RDC: {{".*clang.*"}} "-cc1" "-triple" "amdgpu9.00-amd-amdhsa" {{.*}} "-emit-llvm-bc" {{.*}} "-o" "hip-save-temps-hip-amdgpu9.00-amd-amdhsa-gfx900.{{(bc|o)}}"
// CHECK-NOT: "{{.*}}llvm-link"
// CHECK-NOT: "{{.*}}opt"
// CHECK-NOT: "{{.*}}llc"
// NORDC: {{.*lld.*}}"-o" "hip-save-temps-hip-amdgpu9.00-amd-amdhsa-gfx900.out"
// RDCL: "{{.*lld.*}}" {{.*}} "-plugin-opt=-amdgpu-internalize-symbols" {{.*}}"-save-temps"
// RDCL-SAME: "-o" "a.out-hip-amdgpu9.00-amd-amdhsa-gfx900"
// RDCC: "{{.*clang.*}}" "-cc1" {{.*}} "-E" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.hipi"
// RDCC: "{{.*clang.*}}" "-cc1" {{.*}} "-emit-llvm-bc" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.bc"
// RDCC: "{{.*clang.*}}" "-cc1" {{.*}} "-S" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.s"
// RDCC: "{{.*clang.*}}" "-cc1as" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.o"
// RDCC: "{{.*clang-offload-bundler.*}}" {{.*}} "-output=hip-save-temps.o"
// RDCL: "{{.*clang-offload-bundler.*}}" {{.*}} "-output=hip-save-temps-hip-amdgpu-amd-amdhsa.hipfb"
// RDCL: "{{.*clang.*}}" "-target" "x86_64-unknown-linux-gnu" "-o" "{{.*}}.o" "-x" "assembler" "{{.*}}.mcin" "-c"

// -fno-gpu-rdc host object path
// NORDC: "{{.*clang.*}}" "-cc1" {{.*}} "-E" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.hipi"
// NORDC: "{{.*clang.*}}" "-cc1" {{.*}} "-emit-llvm-bc" {{.*}}  "-fcuda-include-gpubinary" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.bc"
// NORDC: "{{.*clang.*}}" "-cc1" {{.*}} "-S" {{.*}} "-o" "hip-save-temps-host-x86_64-unknown-linux-gnu.s"
// NORDC: "{{.*clang.*}}" "-cc1as" {{.*}} "-o" "hip-save-temps{{.*}}.o"

// output to default a.out or -o specified file name
// NOUT: {{.*}}ld{{.*}}"-o" "a.out"
// WOUT: {{.*}}ld{{.*}}"-o" "executable"

// Check to ensure we have a .tmp for unoptimized bitcode with -fno-gpu-rdc.
// RUN: %clang -### -ccc-print-bindings --target=x86_64-linux-gnu -nogpulib \
// RUN:   -save-temps -fno-gpu-rdc -nogpuinc --offload-arch=gfx906 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=NORDC-BINDINGS %s
//      NORDC-BINDINGS: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI:.+]]"
// NORDC-BINDINGS-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[HIPI]]"], output: "[[TMP_BC:.+]]"
// NORDC-BINDINGS-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[TMP_BC]]"], output: "[[BC:.+]]"
// NORDC-BINDINGS-NEXT: "amdgpu-amd-amdhsa" - "Offload::Packager", inputs: ["[[BC]]"], output: "[[OUTPUT:.+]]"
// NORDC-BINDINGS-NEXT: "amdgpu-amd-amdhsa" - "Offload::Linker", inputs: ["[[OUTPUT]]"], output: "[[FATBIN:.+]]"




// Check to ensure we have a .tmp for unoptimized bitcode with -fgpu-rdc.
// RUN: %clang -### -ccc-print-bindings --target=x86_64-linux-gnu -nogpulib \
// RUN:   -save-temps -fgpu-rdc -nogpuinc --offload-arch=gfx906 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=RDC-BINDINGS %s
//      RDC-BINDINGS: # "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI:.+]]"
// RDC-BINDINGS-NEXT: # "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[HIPI]]"], output: "[[TMP_BC:.+]]"
// RDC-BINDINGS-NEXT: # "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[TMP_BC]]"], output: "[[BC:.+]]"
// RDC-BINDINGS-NEXT: # "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[BC]]"], output: "[[OUTPUT:.+]]"



// RUN: %clang -### -ccc-print-bindings --target=x86_64-linux-gnu -nogpulib \
// RUN:   -save-temps -fno-gpu-rdc -nogpuinc --offload-arch=gfx906 --offload-arch=gfx1030 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=NORDC-BINDINGS-MULTIARCH %s
//
//      NORDC-BINDINGS-MULTIARCH: "amdgpu10.30-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI_0:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu10.30-amd-amdhsa" - "clang", inputs: ["[[HIPI_0]]"], output: "[[TMP_BC_0:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu10.30-amd-amdhsa" - "clang", inputs: ["[[TMP_BC_0]]"], output: "[[BC_0:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI_1:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[HIPI_1]]"], output: "[[TMP_BC_1:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[TMP_BC_1]]"], output: "[[BC_1:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu-amd-amdhsa" - "Offload::Packager", inputs: ["[[BC_0]]", "[[BC_1]]"], output: "[[OUTPUT:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu-amd-amdhsa" - "Offload::Linker", inputs: ["[[OUTPUT]]"], output: "[[FATBIN:.+]]"
