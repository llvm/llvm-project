// RUN: %clang -### --target=amdgcn--amdhsa -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -g --target=amdgcn--amdhsa -mcpu=kaveri -nogpulib %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// RUN: %clang -### --target=amdgcn-amd-amdpal -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -g --target=amdgcn-amd-amdpal -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// RUN: %clang -### --target=amdgcn-mesa-mesa3d -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -g --target=amdgcn-mesa-mesa3d -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s

// AS_LINK: "-cc1as"
// AS_LINK: ld.lld{{.*}} "--no-undefined" "-shared"

// DWARF_VER: "-dwarf-version=5"

// RUN: %clang -### --target=amdgcn--amdhsa -x assembler \
// RUN:  -Wl,--unresolved-symbols=ignore-all %s 2>&1 | FileCheck -check-prefix=AS_LINK_UR %s
// RUN: %clang -### --target=amdgcn--amdhsa -x assembler \
// RUN:  -Xlinker --unresolved-symbols=ignore-all %s 2>&1 | FileCheck -check-prefix=AS_LINK_UR %s

// AS_LINK_UR: "-cc1as"
// AS_LINK_UR: ld.lld{{.*}} "--no-undefined"{{.*}} "--unresolved-symbols=ignore-all"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+:sramecc- -nogpulib \
// RUN:   -L. -flto -fconvergent-functions %s 2>&1 | FileCheck -check-prefix=LTO %s
// LTO: clang{{.*}} "-flto=full"{{.*}}"-fconvergent-functions"
// LTO: ld.lld{{.*}}"-L."{{.*}}"-plugin-opt=mcpu=gfx90a"{{.*}}"--lto-partitions={{[0-9]+}}"{{.*}}"-plugin-opt=-mattr=-sramecc,+xnack"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+:sramecc- -nogpulib \
// RUN:   -L. -fconvergent-functions %s 2>&1 | FileCheck -check-prefix=MCPU %s
// MCPU: ld.lld{{.*}}"-L."{{.*}}"-plugin-opt=mcpu=gfx90a"{{.*}}"-plugin-opt=-mattr=-sramecc,+xnack"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906 -nogpulib \
// RUN:   -fuse-ld=ld %s 2>&1 | FileCheck -check-prefixes=LD %s
// LD: ld.lld

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906 -nogpulib \
// RUN:   -r %s 2>&1 | FileCheck -check-prefixes=RELO %s
// RELO-NOT: -shared

// RUN: %clang -target amdgcn-amd-amdhsa -march=gfx90a -stdlib -startfiles \
// RUN:   -nogpulib -nogpuinc -### %s 2>&1 | FileCheck -check-prefix=STARTUP %s
// STARTUP: ld.lld{{.*}}"-lc" "-lm" "{{.*}}crt1.o"

// Check --flto-partitions

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -nogpulib \
// RUN:   -L. -flto --flto-partitions=42 %s 2>&1 | FileCheck -check-prefix=LTO_PARTS %s
// LTO_PARTS: ld.lld{{.*}}"-L."{{.*}}"-plugin-opt=mcpu=gfx90a"{{.*}}"--lto-partitions=42"

// RUN: not %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -nogpulib \
// RUN:   -L. -flto --flto-partitions=a %s 2>&1 | FileCheck -check-prefix=LTO_PARTS_INV0 %s
// LTO_PARTS_INV0: clang: error: invalid integral value 'a' in '--flto-partitions=a'

// RUN: not %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a -nogpulib \
// RUN:   -L. -flto --flto-partitions=0 %s 2>&1 | FileCheck -check-prefix=LTO_PARTS_INV1 %s
// LTO_PARTS_INV1: clang: error: invalid integral value '0' in '--flto-partitions=0'
