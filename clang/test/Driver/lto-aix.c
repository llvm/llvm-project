// Test LTO path, mcpu and opt level options
// RUN: %clang --target=powerpc-ibm-aix -### %s -flto -fuse-ld=ld -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=LTOPATH,MCPUOPTLEVEL %s
//
// LTOPATH: "-bplugin:{{.*}}libLTO.{{so|dll|dylib}}"
// MCPUOPTLEVEL: "-bplugin_opt:-mcpu={{.*}}" "-bplugin_opt:-O3"
//
// More opt level option tests
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -O -### 2>&1 | FileCheck --check-prefix=O1 %s
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -O1 -### 2>&1 | FileCheck --check-prefix=O1 %s
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -Og -### 2>&1 | FileCheck --check-prefix=O1 %s
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -O2 -### 2>&1 | FileCheck --check-prefix=O2 %s
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -Os -### 2>&1 | FileCheck --check-prefix=O2 %s
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -Oz -### 2>&1 | FileCheck --check-prefix=O2 %s
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -O3 -### 2>&1 | FileCheck --check-prefix=O3 %s
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -Ofast -### 2>&1 | FileCheck --check-prefix=O3 %s
//
// O1: "-bplugin_opt:-O1"
// O2: "-bplugin_opt:-O2"
// O3: "-bplugin_opt:-O3"
//
// vec-extabi option
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -mabi=vec-extabi -### 2>&1 \
// RUN:   | FileCheck --check-prefix=VECEXTABI %s
// RUN: %clang --target=powerpc-ibm-aix --sysroot %S/Inputs/aix_ppc_tree %s \
// RUN:   -fuse-ld=ld -flto -### 2>&1 | FileCheck --check-prefix=NOVECEXTABI %s
//
// VECEXTABI: "-bplugin_opt:-vec-extabi"
// NOVECEXTABI-NOT: "-bplugin_opt:-vec-extabi"
//
// Test debugging options
// RUN: %clang --target=powerpc-ibm-aix -### %s -flto -fuse-ld=ld -gdbx 2>&1 \
// RUN:   | FileCheck -check-prefix=DBX %s
// RUN: %clang --target=powerpc-ibm-aix -### %s -flto -fuse-ld=ld -g 2>&1 \
// RUN:   | FileCheck -check-prefix=NODEBUGGER-TUNE %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### %s -flto -g 2>&1 \
// RUN:   | FileCheck -check-prefixes=STRICT,NODEBUGGER-TUNE %s
// RUN: %clang --target=powerpc64-ibm-aix-xcoff -### %s -flto -g 2>&1 \
// RUN:   | FileCheck -check-prefixes=STRICT,NODEBUGGER-TUNE %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### %s -flto -g -gdbx 2>&1 \
// RUN:   | FileCheck -check-prefix=DBX -check-prefix=STRICT %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### %s -flto -g -ggdb 2>&1 \
// RUN:   | FileCheck -check-prefix=GDB -check-prefix=STRICT %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### %s -flto -g -ggdb0 2>&1 \
// RUN:   | FileCheck -check-prefix=GDB -check-prefix=NOSTRICT %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### %s -flto -g -ggdb1 2>&1 \
// RUN:   | FileCheck -check-prefix=GDB -check-prefix=STRICT %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### %s -flto -g -g0 2>&1 \
// RUN:   | FileCheck -check-prefix=NOSTRICT %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### %s -flto -g -gno-strict-dwarf 2>&1 \
// RUN:   | FileCheck -check-prefix=NOSTRICT %s
// RUN: %clang --target=powerpc-ibm-aix-xcoff -### %s -flto -gstrict-dwarf 2>&1 \
// RUN:   | FileCheck -check-prefix=NOSTRICT %s
//
// DBX:    "-bplugin_opt:-debugger-tune=dbx"
// GDB:    "-bplugin_opt:-debugger-tune=gdb"
// NODEBUGGER-TUNE-NOT: "-bplugin_opt:-debugger-tune="
//
// STRICT:       "-bplugin_opt:-strict-dwarf=true"
// NOSTRICT-NOT: "-bplugin_opt:-strict-dwarf=true"
