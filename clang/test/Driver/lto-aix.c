// Test LTO path, mcpu and opt level options
// RUN: %clang --target=powerpc-ibm-aix -### %s -flto -fuse-ld=ld -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=LTOPATH,MCPUOPTLEVEL %s
//
// LTOPATH: "-bplugin:{{.*}}libLTO.{{so|dll|dylib}}"
// MCPUOPTLEVEL: "-bplugin_opt:-mcpu={{.*}}" "-bplugin_opt:-O3"
//
// Test debugging options
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
