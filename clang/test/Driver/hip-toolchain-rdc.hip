// RUN: %clang -### --target=x86_64-linux-gnu -v \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -fgpu-rdc -nogpuinc \
// RUN:   --no-offload-new-driver -fhip-dump-offload-linker-script \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,LNX %s

// RUN: %clang -### --target=x86_64-pc-windows-msvc -v \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib=lib1.bc --hip-device-lib=lib2.bc \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib1 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_multiple_inputs/lib2 \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -fgpu-rdc -nogpuinc \
// RUN:   --no-offload-new-driver -fhip-dump-offload-linker-script \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck -check-prefixes=CHECK,MSVC %s

// Test fatbin symbol search for -l libraries.

// RUN: touch librdctest.a rdctest2.bin
// RUN: %clang -### --target=x86_64-linux-gnu -v \
// RUN:   -fgpu-rdc -nogpuinc -nogpulib \
// RUN:   --no-offload-new-driver -L. -lrdctest -l:rdctest2.bin \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -check-prefixes=LIB,LNX-LIB %s
// RUN: rm librdctest.a rdctest2.bin

// RUN: touch rdctest.lib rdctest2.bin
// RUN: %clang -### --target=x86_64-pc-windows-msvc -v \
// RUN:   -fgpu-rdc -nogpuinc -nogpulib \
// RUN:   --no-offload-new-driver -L. -lrdctest -l:rdctest2.bin \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -check-prefixes=LIB,MSVC-LIB %s
// RUN: rm rdctest.lib rdctest2.bin

// LIB: HIP fatbin symbol search uses library path: .
// LIB: HIP fatbin symbol search found library: .{{/|\\}}rdctest2.bin
// LNX-LIB: HIP fatbin symbol search found library: .{{/|\\}}librdctest.a
// MSVC-LIB: HIP fatbin symbol search found library: .{{/|\\}}rdctest.lib

// check HIP fatbin and gpubin handle symbols and code object alignment in dumped llvm-mc input
// CHECK: Found undefined HIP fatbin symbol: __hip_fatbin_[[ID1:[0-9a-f]+]]
// CHECK: Found undefined HIP fatbin symbol: __hip_fatbin_[[ID2:[0-9a-f]+]]
// CHECK: Found undefined HIP gpubin handle symbol: __hip_gpubin_handle_[[ID1]]
// CHECK: Found undefined HIP gpubin handle symbol: __hip_gpubin_handle_[[ID2]]
// LNX:  .protected __hip_gpubin_handle_[[ID1]]
// LNX:  .type __hip_gpubin_handle_[[ID1]]
// LNX-LABEL:  .section .hip_gpubin_handle,"aw"
// MSVC-LABEL: .section .hip_gpubin_handle,"dw"
// CHECK:  .globl __hip_gpubin_handle_[[ID1]]
// CHECK-NEXT:  .p2align 3
// CHECK-NEXT:__hip_gpubin_handle_[[ID1]]:
// CHECK-NEXT:  .zero 8
// CHECK-NEXT:  .globl __hip_gpubin_handle_[[ID2]]
// CHECK-NEXT:  .set __hip_gpubin_handle_[[ID2]],__hip_gpubin_handle_[[ID1]]
// LNX: .protected __hip_fatbin_[[ID1]]
// LNX: .type __hip_fatbin_[[ID1]],@object
// LNX-LABEL: .section .hip_fatbin,"a",@progbits
// MSVC-LABEL: .section .hip_fatbin,"dw"
// CHECK: .globl __hip_fatbin_[[ID1]]
// CHECK-NEXT: .p2align 12
// CHECK-NEXT:  .globl __hip_fatbin_[[ID2]]
// CHECK-NEXT:  .set __hip_fatbin_[[ID2]],__hip_fatbin_[[ID1]]
// CHECK-NEXT: __hip_fatbin_[[ID1]]:
// CHECK-NEXT: .incbin "[[BUNDLE:.*hipfb]]"
// LNX: .section .note.GNU-stack, "", @progbits
// MSVC-NOT: .note.GNU-stack

// emit objects for host side path
// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" [[HOST:"x86_64-[^"]+"]]
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" [[A_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// CHECK: [[CLANG]] "-cc1" "-triple" [[HOST]]
// CHECK-SAME: "-aux-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" [[B_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// generate image for device side path on gfx803
// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" [[HOST:"x86_64-[^"]+"]]
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-fcuda-allow-variadic-functions" "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" [[A_BC1:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" [[HOST]]
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device" "-fno-threadsafe-statics" "-fcuda-allow-variadic-functions" "-fvisibility=hidden"
// CHECK-SAME: "-fapply-global-visibility-to-externs"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx803"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" [[B_BC1:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK-NOT: "*.llvm-link"
// CHECK-NOT: ".*opt"
// CHECK-NOT: ".*llc"
// CHECK: {{".*lld.*"}} {{.*}} "-plugin-opt=-amdgpu-internalize-symbols"
// CHECK-SAME: "-plugin-opt=mcpu=gfx803"
// CHECK-SAME: "-o" "[[IMG_DEV1:.*.out]]" [[A_BC1]] [[B_BC1]]

// generate image for device side path on gfx900
// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" [[HOST]]
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "a.cu"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: {{.*}} "-o" [[A_BC2:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[CLANG]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" [[HOST]]
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: {{.*}} "-main-file-name" "b.hip"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "{{.*}}lib1.bc" "{{.*}}lib2.bc"
// CHECK-SAME: "-target-cpu" "gfx900"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" [[B_BC2:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// CHECK-NOT: "*.llvm-link"
// CHECK-NOT: ".*opt"
// CHECK-NOT: ".*llc"
// CHECK: {{".*lld.*"}} {{.*}} "-plugin-opt=-amdgpu-internalize-symbols"
// CHECK-SAME: "-plugin-opt=mcpu=gfx900"
// CHECK-SAME: "-o" "[[IMG_DEV2:.*.out]]" [[A_BC2]] [[B_BC2]]

// combine images generated into hip fat binary object
// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o"
// CHECK-SAME: "-bundle-align=4096"
// CHECK-SAME: "-targets={{.*}},hipv4-amdgcn-amd-amdhsa--gfx803,hipv4-amdgcn-amd-amdhsa--gfx900"
// CHECK-SAME: "-input={{.*}}" "-input=[[IMG_DEV1]]" "-input=[[IMG_DEV2]]" "-output=[[BUNDLE]]"

// CHECK: [[MC:".*clang.*"]] "-target" [[HOST]] "-o" [[OBJBUNDLE:".*o"]] "{{.*}}.mcin"

// output the executable
// LNX: [[LD:".*ld.*"]] {{.*}}"-o" "a.out" {{.*}} [[A_OBJ_HOST]] [[B_OBJ_HOST]] [[OBJBUNDLE]]
// MSVC: [[LD:".*lld-link.*"]] {{.*}}"-out:a.exe" {{.*}} [[A_OBJ_HOST]] [[B_OBJ_HOST]] [[OBJBUNDLE]]
