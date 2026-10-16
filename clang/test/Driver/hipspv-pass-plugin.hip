// REQUIRES: spirv-registered-target
// UNSUPPORTED: system-windows

// RUN: touch %t.dummy.o
// RUN: env "PATH=" %clang -### --no-default-config -o /dev/null --target=spirv64-unknown-chipstar \
// RUN:   %t.dummy.o --hip-path=%S/Inputs/hipspv \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHIPSTAR,FROM-HIP-PATH

// RUN: env "PATH=" %clang -### --no-default-config -o /dev/null --target=spirv64-unknown-chipstar \
// RUN:   %t.dummy.o --hipspv-pass-plugin=%S/Inputs/pass-plugin.so \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHIPSTAR,FROM-OPTION

// RUN: not env "PATH=" %clang -### --no-default-config -o /dev/null --target=spirv64-unknown-chipstar \
// RUN:   %t.dummy.o --hipspv-pass-plugin=foo.so \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHIPSTAR,FROM-OPTION-INVALID

// RUN: env "PATH=" %clang -### --no-default-config -o /dev/null --target=spirv64-unknown-chipstar \
// RUN:   %t.dummy.o 2>&1 | FileCheck %s --check-prefixes=CHIPSTAR,NO-PLUGIN

// FROM-HIP-PATH: {{".*opt"}} {{".*.bc"}} "-load-pass-plugin"
// FROM-HIP-PATH-SAME: {{".*/Inputs/hipspv/lib/libLLVMHipSpvPasses.so"}}
// FROM-OPTION: {{".*opt"}} {{".*.bc"}} "-load-pass-plugin"
// FROM-OPTION-SAME: {{".*/Inputs/pass-plugin.so"}}
// FROM-OPTION-INVALID: error: no such file or directory: 'foo.so'
// NO-PLUGIN-NOT: {{".*opt"}} {{".*.bc"}} "-load-pass-plugin"
// NO-PLUGIN-NOT: {{".*/Inputs/hipspv/lib/libLLVMHipSpvPasses.so"}}
// CHIPSTAR: {{".*clang.*"}} "-cc1"
// CHIPSTAR-SAME: "-emit-obj"
