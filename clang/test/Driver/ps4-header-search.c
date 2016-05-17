// REQUIRES: x86-registered-target

// RUN: env SCE_ORBIS_SDK_DIR=%S/Inputs/scei-ps4_tree %clang -target x86_64-scei-ps4 -E -v %s 2>&1 | FileCheck %s --check-prefix=ENVPS4
// ENVPS4: Inputs/scei-ps4_tree/target/include{{$}}
// ENVPS4: Inputs/scei-ps4_tree/target/include_common{{$}}

// RUN: %clang -isysroot %S/Inputs/scei-ps4_tree -target x86_64-scei-ps4 -E -v %s 2>&1 | FileCheck %s --check-prefix=SYSROOTPS4
// SYSROOTPS4: "{{[^"]*}}clang{{[^"]*}}"
// SYSROOTPS4: Inputs/scei-ps4_tree/target/include{{$}}
// SYSROOTPS4: Inputs/scei-ps4_tree/target/include_common{{$}}
