// REQUIRES: amdgpu-registered-target

// Test -foffload-object-linking flag behavior for HIP.

//--- Device compilation: -foffload-object-linking passes -mllvm -amdgpu-enable-object-linking
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-object-linking \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=MLLVM %s
// MLLVM: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"
// MLLVM-SAME: "-emit-obj"
// MLLVM-NOT: "-flto
// MLLVM-SAME: "-mllvm" "-amdgpu-enable-object-linking"

//--- Device compilation: without -foffload-object-linking, no -amdgpu-enable-object-linking
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=NO-MLLVM %s
// NO-MLLVM: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"
// NO-MLLVM-NOT: "-amdgpu-enable-object-linking"

//--- Device compilation: -fno-offload-object-linking overrides -foffload-object-linking
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc \
// RUN:   -foffload-object-linking -fno-offload-object-linking \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=NEGATED %s
// NEGATED: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"
// NEGATED-NOT: "-amdgpu-enable-object-linking"

//--- Link step: -foffload-object-linking does not forward -flto to linker wrapper
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-object-linking \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NO-LTO %s
// NO-LTO: clang-linker-wrapper
// NO-LTO-NOT: "--device-compiler=amdgpu-amd-amdhsa=-flto"
// NO-LTO-NOT: "--device-compiler=amdgpu-amd-amdhsa=-flto-partitions=

//--- Link step: without -foffload-object-linking, -flto is forwarded
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc \
// RUN:   %s 2>&1 | FileCheck -check-prefix=LTO-DEFAULT %s
// LTO-DEFAULT: clang-linker-wrapper
// LTO-DEFAULT-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=full"

//--- Link step: -foffload-object-linking invokes clang-linker-wrapper for .o inputs
// RUN: touch %t1.o %t2.o
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-object-linking \
// RUN:   %t1.o %t2.o 2>&1 | FileCheck -check-prefix=WRAPPER %s
// WRAPPER: clang-linker-wrapper
// WRAPPER-SAME: "--should-extract=gfx90a"

//--- Host compilation: -foffload-object-linking does not affect host cc1
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-object-linking \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=HOST %s
// HOST: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST-NOT: "-amdgpu-enable-object-linking"
// HOST-SAME: "-emit-obj"

//--- Error: -foffload-object-linking conflicts with full -foffload-lto
// RUN: not %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-object-linking \
// RUN:   -foffload-lto %s 2>&1 | FileCheck -check-prefix=CONFLICT %s
// CONFLICT: error: cannot specify '-foffload-lto' along with '-foffload-object-linking'

//--- ThinLTO implies object linking; an explicit object-linking flag is redundant
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-object-linking \
// RUN:   -foffload-lto=thin -c %s 2>&1 | FileCheck -check-prefix=THIN-OBJECT %s
// THIN-OBJECT: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"
// THIN-OBJECT-SAME: "-emit-llvm-bc"
// THIN-OBJECT-SAME: "-flto=thin"
// THIN-OBJECT-SAME: "-mllvm" "-amdgpu-enable-object-linking"

//--- Link step: ThinLTO forwards object-linking mode to the LTO backend.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-lto=thin %s 2>&1 \
// RUN:   | FileCheck -check-prefix=THIN-LINK %s
// THIN-LINK: clang-linker-wrapper
// THIN-LINK-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"
// THIN-LINK-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=thin"
