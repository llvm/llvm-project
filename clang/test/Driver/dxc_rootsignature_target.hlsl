// RUN: %clang_dxc -E EntryRS -T rootsig_1_1 /Fo %t.dxo -### %s 2>&1 | FileCheck %s --check-prefix=CMDS

// RUN: %clang_dxc -E EntryRS -T rootsig_1_1 /Fo %t.dxo %s
// RUN: obj2yaml %t.dxo | FileCheck %s --check-prefix=OBJ

// CMDS: "{{.*}}clang{{.*}}" "-cc1"
// CMDS-SAME: "-triple" "dxilv1.1-unknown-shadermodel1.1-rootsignature"
// CMDS-SAME: "-hlsl-entry" "EntryRS"
// CMDS: "{{.*}}llvm-objcopy{{(.exe)?}}" "{{.*}}.dxo" "--only-section=RTS0"

#define EntryRS "UAV(u0)"

// OBJ: --- !dxcontainer
// FileSize = 32 (header) + 48 (RTS0 content) + 4 (1 part offset) + 8 (1 part header)
// OBJ:       FileSize: 92
// OBJ-NEXT:  PartCount:        1
// OBJ-NEXT:  PartOffsets:      [ 36 ]
// OBJ-NEXT:  Parts:
// OBJ-NOT:   DXIL
// OBJ-NOT:   SFI0
// OBJ-NOT:   HASH
// OBJ-NOT:   ISG0
// OBJ-NOT:   OSG0

// OBJ:       - Name: RTS0
// OBJ-NEXT:    Size:         48
// OBJ-NEXT:    RootSignature:
// OBJ-NEXT:      Version:         2
// OBJ-NEXT:      NumRootParameters: 1
// OBJ-NEXT:      RootParametersOffset: 24

// OBJ:         Parameters:
// UAV(u0)
// OBJ:          - ParameterType:   4
// OBJ-NEXT:       ShaderVisibility: 0
// OBJ-NEXT:       Descriptor:
// OBJ-NEXT:         RegisterSpace:   0
// OBJ-NEXT:         ShaderRegister:  0

// OBJ-NOT: PSV0
