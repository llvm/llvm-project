// RUN: %clang -### -target amdgcn--amdhsa -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -g -target amdgcn--amdhsa -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// RUN: %clang -### -target amdgcn-amd-amdpal -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -g -target amdgcn-amd-amdpal -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// RUN: %clang -### -target amdgcn-mesa-mesa3d -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -g -target amdgcn-mesa-mesa3d -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s

// AS_LINK: clang{{.*}} "-cc1as"
// AS_LINK: ld.lld{{.*}} "-shared"

// DWARF_VER: "-dwarf-version=5"

// RUN: %clang -### -target amdgcn-amd-amdhsa -mcpu=gfx906 -nogpulib \
// RUN:   -flto %s 2>&1 | FileCheck -check-prefix=LTO %s
// LTO: clang{{.*}} "-flto"
// LTO: ld.lld{{.*}}

// TODO: Remove during upstreaming target id.
// RUN: %clang -### -target amdgcn--amdhsa -mcpu=gfx900 -mcode-object-v3 %s 2>&1 | FileCheck -check-prefix=COV3 %s
// COV3: clang{{.*}} "-mllvm" "--amdhsa-code-object-version=3"
// RUN: %clang -### -target amdgcn--amdhsa -mcpu=gfx900 -mno-code-object-v3 %s 2>&1 | FileCheck -check-prefix=COV2 %s
// COV2: clang{{.*}} "-mllvm" "--amdhsa-code-object-version=2"
