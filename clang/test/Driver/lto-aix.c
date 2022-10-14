// Test LTO path, mcpu and opt level options
// RUN: %clang --target=powerpc-ibm-aix -### %s -flto -fuse-ld=ld -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=LTOPATH,MCPUOPTLEVEL %s
//
// LTOPATH: "-bplugin:{{.*}}libLTO.{{so|dll|dylib}}"
// MCPUOPTLEVEL: "-bplugin_opt:-mcpu={{.*}}" "-bplugin_opt:-O3"
