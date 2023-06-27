// RUN: %clang -### --target=amdgcn--amdhsa -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -g --target=amdgcn--amdhsa -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// RUN: %clang -### --target=amdgcn-amd-amdpal -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -g --target=amdgcn-amd-amdpal -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// RUN: %clang -### --target=amdgcn-mesa-mesa3d -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -g --target=amdgcn-mesa-mesa3d -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s

// AS_LINK: "-cc1as"
// AS_LINK: ld.lld{{.*}} "--no-undefined" "-shared"

// DWARF_VER: "-dwarf-version=5"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906 -nogpulib \
// RUN:   -L. -flto -fconvergent-functions %s 2>&1 | FileCheck -check-prefixes=LTO,MCPU %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906 -nogpulib \
// RUN:   -L. -fconvergent-functions %s 2>&1 | FileCheck -check-prefix=MCPU %s
// LTO: clang{{.*}} "-flto=full"{{.*}}"-fconvergent-functions"
// MCPU: ld.lld{{.*}}"-L."{{.*}}"-plugin-opt=mcpu=gfx906"
