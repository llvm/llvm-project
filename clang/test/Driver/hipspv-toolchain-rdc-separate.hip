// UNSUPPORTED: system-windows

// RUN: %clang -### -x hip -target x86_64-linux-gnu \
// RUN:   --offload=spirv64-unknown-chipstar --offload-new-driver -fgpu-rdc -c \
// RUN:   --hip-path=%S/Inputs/hipspv -nohipwrapperinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck  \
// RUN:   -DHIP_PATH=%S/Inputs/hipspv %s

// CHECK: [[CLANG:".*clang[^ ]*"]] "-cc1" "-triple" "spirv64-unknown-chipstar"
// CHECK-SAME: "-aux-triple" "[[HOST_TRIPLE:[^ ]*]]"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "-fvisibility=hidden" "-fapply-global-visibility-to-externs"
// CHECK-SAME: "-mlink-builtin-bitcode" "[[HIP_PATH]]/lib/hip-device-lib/hipspv-spirv64-unknown-chipstar.bc"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: "-o" "[[A_DEV_BC:.*bc]]" "-x" "hip"
// CHECK-SAME: "[[A_SRC:.*a.cu]]"

// CHECK: "{{.*llvm-offload-binary[^ ]*}}" "-o" "[[A_BIN_PACKAGE:.*.out]]"
// CHECK-SAME: "--image=file=[[A_DEV_BC]],triple=spirv64-unknown-chipstar,arch=generic,kind=hip"

// CHECK: [[CLANG]] "-cc1" "-triple" "[[HOST_TRIPLE]]"
// CHECK-SAME: "-aux-triple" "spirv64-unknown-chipstar"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: "-fembed-offload-object=[[A_BIN_PACKAGE]]"
// CHECK-SAME: "-o" "[[A_HOST_OBJ:.*o]]" "-x" "hip" "[[A_SRC]]"

// CHECK: [[CLANG]] "-cc1" "-triple" "spirv64-unknown-chipstar"
// CHECK-SAME: "-aux-triple" "[[HOST_TRIPLE]]"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "-fvisibility=hidden" "-fapply-global-visibility-to-externs"
// CHECK-SAME: "-mlink-builtin-bitcode" "[[HIP_PATH]]/lib/hip-device-lib/hipspv-spirv64-unknown-chipstar.bc"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: "-o" "[[B_DEV_BC:.*bc]]" "-x" "hip"
// CHECK-SAME: "[[B_SRC:.*b.hip]]"

// CHECK: "{{.*llvm-offload-binary[^ ]*}}" "-o" "[[B_BIN_PACKAGE:.*.out]]"
// CHECK-SAME: "--image=file=[[B_DEV_BC]],triple=spirv64-unknown-chipstar,arch=generic,kind=hip"

// CHECK: [[CLANG]] "-cc1" "-triple" "[[HOST_TRIPLE]]"
// CHECK-SAME: "-aux-triple" "spirv64-unknown-chipstar"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: "-fembed-offload-object=[[B_BIN_PACKAGE]]"
// CHECK-SAME: "-o" "[[B_HOST_OBJ:.*o]]" "-x" "hip"
// CHECK-SAME: "[[B_SRC]]"

// RUN: rm -rf %t && mkdir %t
// RUN: touch %t/a.o %t/b.o
// RUN: %clang -### -target x86_64-linux-gnu --offload-new-driver -fgpu-rdc \
// RUN:   -Xoffload-compiler-spirv64-unknown-chipstar \
// RUN:   --hip-path=%S/Inputs/hipspv \
// RUN:   -no-hip-rt %t/a.o %t/b.o \
// RUN: 2>&1 | FileCheck -check-prefixes=LINK \
// RUN:   -DHIP_PATH=%S/Inputs/hipspv %s

// LINK: "{{.*clang-linker-wrapper[^ ]*}}"
// LINK-SAME: "--host-triple=x86_64-unknown-linux-gnu"
// LINK-SAME: "--device-compiler=spirv64-unknown-chipstar=--hip-path=[[HIP_PATH]]"
// LINK-SAME: "-o" "a.out"
// LINK-SAME: "{{.*a[.]o}}" "{{.*b[.]o}}"
// LINK-NOT: -lamdhip64
