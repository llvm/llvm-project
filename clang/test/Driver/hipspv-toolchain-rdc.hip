// UNSUPPORTED: system-windows

// RUN: %clang -### -x hip -target x86_64-linux-gnu \
// RUN:   --offload=spirv64-unknown-chipstar -fgpu-rdc \
// RUN:   --hip-path=%S/Inputs/hipspv -nohipwrapperinc -no-hip-rt \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefix=NEW \
// RUN:   -DOFFLOAD_TRIPLE=spirv64-unknown-chipstar -DHIP_PATH=%S/Inputs/hipspv %s

// NEW: [[CLANG:".*clang[^ ]*"]] "-cc1" "-triple" "[[OFFLOAD_TRIPLE]]"
// NEW-SAME: "-aux-triple" "[[HOST_TRIPLE:[^ ]*]]"
// NEW-SAME: "-emit-llvm-bc"
// NEW-SAME: "-fcuda-is-device"
// NEW-SAME: "-fvisibility=hidden" "-fapply-global-visibility-to-externs"
// NEW-SAME: "-mlink-builtin-bitcode" "[[HIP_PATH]]/lib/hip-device-lib/hipspv-spirv64-unknown-chipstar.bc"
// NEW-SAME: "-fgpu-rdc"
// NEW-SAME: "-o" "[[A_DEV_OBJ:.*o]]" "-x" "hip"
// NEW-SAME: "[[A_SRC:.*a.cu]]"

// NEW: "{{.*llvm-offload-binary[^ ]*}}" "-o" "[[A_BIN_PACKAGE:.*.out]]"
// NEW-SAME: "--image=file=[[A_DEV_OBJ]],triple=[[OFFLOAD_TRIPLE]],arch=generic,kind=hip"

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
// NEW-SAME: "-o" "[[B_DEV_OBJ:.*o]]" "-x" "hip"
// NEW-SAME: "[[B_SRC:.*b.hip]]"

// NEW: "{{.*llvm-offload-binary[^ ]*}}" "-o" "[[B_BIN_PACKAGE:.*.out]]"
// NEW-SAME: "--image=file=[[B_DEV_OBJ]],triple=[[OFFLOAD_TRIPLE]],arch=generic,kind=hip"

// NEW: [[CLANG]] "-cc1" "-triple" "[[HOST_TRIPLE]]"
// NEW-SAME: "-aux-triple" "[[OFFLOAD_TRIPLE]]"
// NEW-SAME: "-emit-obj"
// NEW-SAME: "-fgpu-rdc"
// NEW-SAME: "-fembed-offload-object=[[B_BIN_PACKAGE]]"
// NEW-SAME: "-o" "[[B_HOST_OBJ:.*o]]" "-x" "hip"
// NEW-SAME: "[[B_SRC]]"

// NEW: "{{.*clang-linker-wrapper[^ ]*}}"
// NEW-SAME: "--device-compiler=[[OFFLOAD_TRIPLE]]=--hip-path=[[HIP_PATH]]"
// NEW-SAME: "--device-compiler=[[OFFLOAD_TRIPLE]]=-flto=full"
// NEW-SAME: "--host-triple=[[HOST_TRIPLE]]"
// NEW-SAME: "-o" "a.out"
// NEW-SAME: "[[A_HOST_OBJ]]" "[[B_HOST_OBJ]]"
// NEW-NOT: -lamdhip64
