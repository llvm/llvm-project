// RUN: %clang_dxc -E EntryRS -T rootsig_1_1 /Fo %t.dxo -### %s 2>&1 | FileCheck %s --check-prefix=CMDS

// CMDS: "{{.*}}clang{{.*}}" "-cc1"
// CMDS-SAME: "-triple" "dxilv1.1-unknown-shadermodel1.1-rootsignature"
// CMDS-SAME: "-hlsl-entry" "EntryRS"
// CMDS: "{{.*}}llvm-objcopy{{(.exe)?}}" "{{.*}}.dxo" "--only-section=RTS0"

#define EntryRS "UAV(u0)"
