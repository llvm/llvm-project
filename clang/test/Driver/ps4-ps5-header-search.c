/// PS4 and PS5 use the same SDK layout, so use the same tree for both.
// RUN: env SCE_ORBIS_SDK_DIR=%S/Inputs/scei-ps4_tree %clang -target x86_64-scei-ps4 -E -v %s 2>&1 | FileCheck %s --check-prefix=ENVPS4
// RUN: env SCE_PROSPERO_SDK_DIR=%S/Inputs/scei-ps4_tree %clang -target x86_64-sie-ps5 -E -v %s 2>&1 | FileCheck %s --check-prefix=ENVPS4
// ENVPS4: Inputs/scei-ps4_tree/target/include
// ENVPS4: Inputs/scei-ps4_tree/target/include_common
// ENVPS4-NOT: /usr/include

// RUN: %clang -isysroot %S/Inputs/scei-ps4_tree -target x86_64-scei-ps4 -E -v %s 2>&1 | FileCheck %s --check-prefix=SYSROOTPS4
// RUN: %clang -isysroot %S/Inputs/scei-ps4_tree -target x86_64-sie-ps5 -E -v %s 2>&1 | FileCheck %s --check-prefix=SYSROOTPS4
// SYSROOTPS4: "{{[^"]*}}clang{{[^"]*}}"
// SYSROOTPS4: Inputs/scei-ps4_tree/target/include
// SYSROOTPS4: Inputs/scei-ps4_tree/target/include_common
// SYSROOTPS4-NOT: /usr/include
