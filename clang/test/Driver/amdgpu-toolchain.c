// RUN: %clang -### -target amdgcn--amdhsa -x assembler -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=AS_LINK %s
// AS_LINK: clang{{.*}} "-cc1as"
// AS_LINK: ld.lld{{.*}} "-shared"

// RUN: %clang -### -g -target amdgcn--amdhsa -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// DWARF_VER: "-dwarf-version=2"
