// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -c -nogpuinc -nogpulib --cuda-device-only -fgpu-rdc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,EMITBC %s

// With `-emit-llvm`, the output should be the same as the aforementioned line
// as `-fgpu-rdc` in HIP implies `-emit-llvm`.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -c -emit-llvm -nogpuinc -nogpulib --cuda-device-only -fgpu-rdc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,EMITBC %s

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -S -nogpuinc -nogpulib --cuda-device-only -fgpu-rdc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,EMITLL %s

// With `-emit-llvm`, the output should be the same as the aforementioned line
// as `-fgpu-rdc` in HIP implies `-emit-llvm`.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -S -emit-llvm -nogpuinc -nogpulib --cuda-device-only -fgpu-rdc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefixes=COMMON,EMITLL %s

// With `-save-temps`, commane lines for each steps are dumped. For assembly
// output, there should 3 steps (preprocessor, compile, and backend) per source
// and per target, totally 12 steps.

// RUN: %clang -### -save-temps -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -S -nogpuinc -nogpulib --cuda-device-only -fgpu-rdc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip --gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefix=SAVETEMP %s

// Check output one file without bundling cause error.

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -S -nogpuinc -nogpulib --cuda-device-only -fgpu-rdc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu -o %t.s --no-gpu-bundle-output \
// RUN: 2>&1 | FileCheck -check-prefix=FAIL %s

// COMMON: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// EMITBC-SAME: "-emit-llvm-bc"
// EMITLL-SAME: "-emit-llvm"
// COMMON-SAME: {{.*}} "-main-file-name" "a.cu"
// COMMON-SAME: "-fcuda-is-device" "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// COMMON-SAME: "-fapply-global-visibility-to-externs"
// COMMON-SAME: "-target-cpu" "gfx803"
// COMMON-SAME: "-fgpu-rdc"
// EMITBC-SAME: {{.*}} "-o" {{".*a.*bc"}} "-x" "hip"
// EMITLL-SAME: {{.*}} "-o" {{".*a.*ll"}} "-x" "hip"
// CHECK-SAME: {{.*}} {{".*a.cu"}}

// COMMON: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// EMITBC-SAME: "-emit-llvm-bc"
// EMITLL-SAME: "-emit-llvm"
// COMMON-SAME: {{.*}} "-main-file-name" "a.cu"
// COMMON-SAME: "-fcuda-is-device" "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// COMMON-SAME: "-fapply-global-visibility-to-externs"
// COMMON-SAME: "-target-cpu" "gfx900"
// COMMON-SAME: "-fgpu-rdc"
// EMITBC-SAME: {{.*}} "-o" {{".*a.*bc"}} "-x" "hip"
// EMITLL-SAME: {{.*}} "-o" {{".*a.*ll"}} "-x" "hip"
// COMMON-SAME: {{.*}} {{".*a.cu"}}

// COMMON: "{{.*}}clang-offload-bundler" "-type={{(bc|ll)}}"
// COMMON-SAME: "-targets=hip-amdgcn-amd-amdhsa--gfx803,hip-amdgcn-amd-amdhsa--gfx900"
// COMMON-SAME: "-outputs=a-hip-amdgcn-amd-amdhsa.{{(bc|ll)}}"

// COMMON: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// EMITBC-SAME: "-emit-llvm-bc"
// EMITLL-SAME: "-emit-llvm"
// COMMON-SAME: {{.*}} "-main-file-name" "b.hip"
// COMMON-SAME: "-fcuda-is-device" "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// COMMON-SAME: "-fapply-global-visibility-to-externs"
// COMMON-SAME: "-target-cpu" "gfx803"
// COMMON-SAME: "-fgpu-rdc"
// EMITBC-SAME: {{.*}} "-o" {{".*b.*bc"}} "-x" "hip"
// EMITLL-SAME: {{.*}} "-o" {{".*b.*ll"}} "-x" "hip"
// COMMON-SAME: {{.*}} {{".*b.hip"}}

// COMMON: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// EMITBC-SAME: "-emit-llvm-bc"
// EMITLL-SAME: "-emit-llvm"
// COMMON-SAME: {{.*}} "-main-file-name" "b.hip"
// COMMON-SAME: "-fcuda-is-device" "-fcuda-allow-variadic-functions" "-fvisibility" "hidden"
// COMMON-SAME: "-fapply-global-visibility-to-externs"
// COMMON-SAME: "-target-cpu" "gfx900"
// COMMON-SAME: "-fgpu-rdc"
// EMITBC-SAME: {{.*}} "-o" {{".*b.*bc"}} "-x" "hip"
// EMITLL-SAME: {{.*}} "-o" {{".*b.*ll"}} "-x" "hip"
// COMMON-SAME: {{.*}} {{".*b.hip"}}

// COMMON: "{{.*}}clang-offload-bundler" "-type={{(bc|ll)}}"
// COMMON-SAME: "-targets=hip-amdgcn-amd-amdhsa--gfx803,hip-amdgcn-amd-amdhsa--gfx900"
// COMMON-SAME: "-outputs=b-hip-amdgcn-amd-amdhsa.{{(bc|ll)}}"

// SAVETEMP: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-E"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx803"
// SAVETEMP-SAME: {{.*}} "-o" [[A_GFX803_CUI:"a.*cui"]] "-x" "hip" {{".*a.cu"}}
// SAVETEMP-NEXT: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-emit-llvm-bc"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx803"
// SAVETEMP-SAME: {{.*}} "-o" [[A_GFX803_TMP_BC:"a.*tmp.bc"]] "-x" "hip-cpp-output" [[A_GFX803_CUI]]
// SAVETEMP-NEXT: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-emit-llvm"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx803"
// SAVETEMP-SAME: {{.*}} "-o" {{"a.*.ll"}} "-x" "ir" [[A_GFX803_TMP_BC]]

// SAVETEMP: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-E"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx900"
// SAVETEMP-SAME: {{.*}} "-o" [[A_GFX900_CUI:"a.*cui"]] "-x" "hip" {{".*a.cu"}}
// SAVETEMP-NEXT: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-emit-llvm-bc"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx900"
// SAVETEMP-SAME: {{.*}} "-o" [[A_GFX900_TMP_BC:"a.*tmp.bc"]] "-x" "hip-cpp-output" [[A_GFX900_CUI]]
// SAVETEMP-NEXT: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-emit-llvm"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "a.cu" {{.*}} "-target-cpu" "gfx900"
// SAVETEMP-SAME: {{.*}} "-o" {{"a.*.ll"}} "-x" "ir" [[A_GFX900_TMP_BC]]

// SAVETEMP: "{{.*}}clang-offload-bundler" "-type=ll"
// SAVETEMP-SAME: "-targets=hip-amdgcn-amd-amdhsa--gfx803,hip-amdgcn-amd-amdhsa--gfx900"
// SAVETEMP-SAME: "-outputs=a-hip-amdgcn-amd-amdhsa.ll"

// SAVETEMP: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-E"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx803"
// SAVETEMP-SAME: {{.*}} "-o" [[B_GFX803_CUI:"b.*cui"]] "-x" "hip" {{".*b.hip"}}
// SAVETEMP-NEXT: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-emit-llvm-bc"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx803"
// SAVETEMP-SAME: {{.*}} "-o" [[B_GFX803_TMP_BC:"b.*tmp.bc"]] "-x" "hip-cpp-output" [[B_GFX803_CUI]]
// SAVETEMP-NEXT: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-emit-llvm"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx803"
// SAVETEMP-SAME: {{.*}} "-o" {{"b.*.ll"}} "-x" "ir" [[B_GFX803_TMP_BC]]

// SAVETEMP: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-E"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx900"
// SAVETEMP-SAME: {{.*}} "-o" [[B_GFX900_CUI:"b.*cui"]] "-x" "hip" {{".*b.hip"}}
// SAVETEMP-NEXT: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-emit-llvm-bc"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx900"
// SAVETEMP-SAME: {{.*}} "-o" [[B_GFX900_TMP_BC:"b.*tmp.bc"]] "-x" "hip-cpp-output" [[B_GFX900_CUI]]
// SAVETEMP-NEXT: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"
// SAVETEMP-SAME: "-emit-llvm"
// SAVETEMP-SAME: {{.*}} "-main-file-name" "b.hip" {{.*}} "-target-cpu" "gfx900"
// SAVETEMP-SAME: {{.*}} "-o" {{"b.*.ll"}} "-x" "ir" [[B_GFX900_TMP_BC]]

// SAVETEMP: "{{.*}}clang-offload-bundler" "-type=ll"
// SAVETEMP-SAME: "-targets=hip-amdgcn-amd-amdhsa--gfx803,hip-amdgcn-amd-amdhsa--gfx900"
// SAVETEMP-SAME: "-outputs=b-hip-amdgcn-amd-amdhsa.ll"

// FAIL: error: cannot specify -o when generating multiple output files
