// UNSUPPORTED: system-windows

// RUN: %clang -### -x hip -target x86_64-linux-gnu --offload=spirv64 \
// RUN:   --no-offload-new-driver -fgpu-rdc --hip-path=%S/Inputs/hipspv -nohipwrapperinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefix=OLD %s

// RUN: %clang -### -x hip -target x86_64-linux-gnu \
// RUN:   --offload=spirv64-unknown-chipstar --offload-new-driver -fgpu-rdc \
// RUN:   --hip-path=%S/Inputs/hipspv -nohipwrapperinc -no-hip-rt \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefix=NEW \
// RUN:   -DOFFLOAD_TRIPLE=spirv64-unknown-chipstar -DHIP_PATH=%S/Inputs/hipspv %s

// Emit objects for host side path
// OLD: [[CLANG:".*clang.*"]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// OLD-SAME: "-aux-triple" "spirv64"
// OLD-SAME: "-emit-obj"
// OLD-SAME: "-fgpu-rdc"
// OLD-SAME: {{.*}} "-o" [[A_OBJ_HOST:".*o"]] "-x" "hip"
// OLD-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// OLD: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// OLD-SAME: "-aux-triple" "spirv64"
// OLD-SAME: "-emit-obj"
// OLD-SAME: "-fgpu-rdc"
// OLD-SAME: {{.*}} "-o" [[B_OBJ_HOST:".*o"]] "-x" "hip"
// OLD-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// Emit code (LLVM BC) for device side path.
// OLD: [[CLANG]] "-cc1" "-triple" "spirv64"
// OLD-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// OLD-SAME: "-emit-llvm-bc"
// OLD-SAME: "-fcuda-is-device"
// OLD-SAME: "-fvisibility=hidden" "-fapply-global-visibility-to-externs"
// OLD-SAME: "-fgpu-rdc"
// OLD-SAME: {{.*}} "-o" [[A_BC1:".*bc"]] "-x" "hip"
// OLD-SAME: {{.*}} [[A_SRC]]

// OLD: [[CLANG]] "-cc1" "-triple" "spirv64"
// OLD-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// OLD-SAME: "-emit-llvm-bc"
// OLD-SAME: "-fcuda-is-device"
// OLD-SAME: "-fvisibility=hidden" "-fapply-global-visibility-to-externs"
// OLD-SAME: "-fgpu-rdc"
// OLD-SAME: {{.*}} "-o" [[B_BC1:".*bc"]] "-x" "hip"
// OLD-SAME: {{.*}} [[B_SRC]]

// Link device code, lower it with HIPSPV passes and emit SPIR-V binary.
// OLD: {{".*llvm-link.*"}} "-o" [[AB_LINK:".*bc"]] [[A_BC1]] [[B_BC1]]
// OLD: {{".*opt.*"}} [[AB_LINK]] "-load-pass-plugin"
// OLD-SAME: "{{.*}}/Inputs/hipspv/lib/libLLVMHipSpvPasses.so"
// OLD-SAME: "-o" [[AB_LOWER:".*bc"]]
// OLD: {{".*llvm-spirv"}} "--spirv-max-version=1.1" "--spirv-ext=+all"
// OLD-SAME: [[AB_LOWER]] "-o" "[[AB_SPIRV:.*out]]"

// Construct fat binary object.
// OLD: [[BUNDLER:".*clang-offload-bundler"]] "-type=o" "-bundle-align=4096"
// OLD-SAME: "-targets={{.*}},hip-spirv64----generic"
// OLD-SAME: "-input=/dev/null" "-input=[[AB_SPIRV]]"
// OLD-SAME: "-output=[[AB_FATBIN:.*hipfb]]"
// OLD: {{".*clang.*"}} "-o" [[OBJBUNDLE:".*o"]] "{{.*}}.mcin"

// Output the executable
// OLD: {{".*ld.*"}} {{.*}}"-o" "a.out" {{.*}} [[A_OBJ_HOST]] [[B_OBJ_HOST]]
// OLD-SAME: [[OBJBUNDLE]]

// NEW: [[CLANG:".*clang[^ ]*"]] "-cc1" "-triple" "[[OFFLOAD_TRIPLE]]"
// NEW-SAME: "-aux-triple" "[[HOST_TRIPLE:[^ ]*]]"
// NEW-SAME: "-emit-llvm-bc"
// NEW-SAME: "-fcuda-is-device"
// NEW-SAME: "-fvisibility=hidden" "-fapply-global-visibility-to-externs"
// NEW-SAME: "-mlink-builtin-bitcode" "[[HIP_PATH]]/lib/hip-device-lib/hipspv-spirv64-unknown-chipstar.bc"
// NEW-SAME: "-fgpu-rdc"
// NEW-SAME: "-o" "[[A_DEV_BC:.*bc]]" "-x" "hip"
// NEW-SAME: "[[A_SRC:.*a.cu]]"

// NEW: "{{.*llvm-offload-binary[^ ]*}}" "-o" "[[A_BIN_PACKAGE:.*.out]]"
// NEW-SAME: "--image=file=[[A_DEV_BC]],triple=[[OFFLOAD_TRIPLE]],arch=generic,kind=hip"

// NEW: [[CLANG]] "-cc1" "-triple" "[[HOST_TRIPLE]]"
// NEW-SAME: "-aux-triple" "[[OFFLOAD_TRIPLE]]"
// NEW-SAME: "-emit-obj"
// NEW-SAME: "-fgpu-rdc"
// NEW-SAME: "-fembed-offload-object=[[A_BIN_PACKAGE]]"
// NEW-SAME: "-o" "[[A_HOST_OBJ:.*o]]" "-x" "hip" "[[A_SRC]]"

// NEW: [[CLANG]] "-cc1" "-triple" "[[OFFLOAD_TRIPLE]]"
// NEW-SAME: "-aux-triple" "[[HOST_TRIPLE]]"
// NEW-SAME: "-emit-llvm-bc"
// NEW-SAME: "-fcuda-is-device"
// NEW-SAME: "-fvisibility=hidden" "-fapply-global-visibility-to-externs"
// NEW-SAME: "-mlink-builtin-bitcode" "[[HIP_PATH]]/lib/hip-device-lib/hipspv-spirv64-unknown-chipstar.bc"
// NEW-SAME: "-fgpu-rdc"
// NEW-SAME: "-o" "[[B_DEV_BC:.*bc]]" "-x" "hip"
// NEW-SAME: "[[B_SRC:.*b.hip]]"

// NEW: "{{.*llvm-offload-binary[^ ]*}}" "-o" "[[B_BIN_PACKAGE:.*.out]]"
// NEW-SAME: "--image=file=[[B_DEV_BC]],triple=[[OFFLOAD_TRIPLE]],arch=generic,kind=hip"

// NEW: [[CLANG]] "-cc1" "-triple" "[[HOST_TRIPLE]]"
// NEW-SAME: "-aux-triple" "[[OFFLOAD_TRIPLE]]"
// NEW-SAME: "-emit-obj"
// NEW-SAME: "-fgpu-rdc"
// NEW-SAME: "-fembed-offload-object=[[B_BIN_PACKAGE]]"
// NEW-SAME: "-o" "[[B_HOST_OBJ:.*o]]" "-x" "hip"
// NEW-SAME: "[[B_SRC]]"

// NEW: "{{.*clang-linker-wrapper[^ ]*}}"
// NEW-SAME: "--device-compiler=[[OFFLOAD_TRIPLE]]=--hip-path=[[HIP_PATH]]"
// NEW-SAME: "--host-triple=[[HOST_TRIPLE]]"
// NEW-SAME: "-o" "a.out"
// NEW-SAME: "[[A_HOST_OBJ]]" "[[B_HOST_OBJ]]"
// NEW-NOT: -lamdhip64
