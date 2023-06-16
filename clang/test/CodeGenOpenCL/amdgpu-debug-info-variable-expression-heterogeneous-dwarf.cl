// REQUIRES: amdgpu-registered-target
// RUN: %clang -Xclang -cl-std=CL2.0 -emit-llvm -fno-discard-value-names -g -gheterogeneous-dwarf -O0 -S -nogpulib -target amdgcn-amd-amdhsa -mcpu=fiji -o - %s | FileCheck %s
// RUN: %clang -Xclang -cl-std=CL2.0 -emit-llvm -fno-discard-value-names -g -gheterogeneous-dwarf -O0 -S -nogpulib -target amdgcn-amd-amdhsa-opencl -mcpu=fiji -o - %s | FileCheck %s

// CHECK-DAG: @FileVar0 = hidden addrspace(1) global ptr addrspace(1) null, align 8, !dbg.def ![[FILEVAR0_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR0_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR0_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR0_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(1))), argObjects: {![[FILEVAR0_F]]})
// CHECK-DAG: ![[FILEVAR0_F]] = distinct !DIFragment()
global int *FileVar0;
// CHECK-DAG: @FileVar1 = hidden addrspace(1) global ptr addrspace(4) null, align 8, !dbg.def ![[FILEVAR1_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR1_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR1_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR1_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(4))), argObjects: {![[FILEVAR1_F]]})
// CHECK-DAG: ![[FILEVAR1_F]] = distinct !DIFragment()
constant int *FileVar1;
// CHECK-DAG: @FileVar2 = hidden addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4, !dbg.def ![[FILEVAR2_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR2_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR2_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR2_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(3))), argObjects: {![[FILEVAR2_F]]})
// CHECK-DAG: ![[FILEVAR2_F]] = distinct !DIFragment()
local int *FileVar2;
// CHECK-DAG: @FileVar3 = hidden addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4, !dbg.def ![[FILEVAR3_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR3_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR3_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR3_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(5))), argObjects: {![[FILEVAR3_F]]})
// CHECK-DAG: ![[FILEVAR3_F]] = distinct !DIFragment()
private int *FileVar3;
// CHECK-DAG: @FileVar4 = hidden addrspace(1) global ptr null, align 8, !dbg.def ![[FILEVAR4_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR4_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR4_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR4_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr)), argObjects: {![[FILEVAR4_F]]})
// CHECK-DAG: ![[FILEVAR4_F]] = distinct !DIFragment()
int *FileVar4;

// CHECK-DAG: @FileVar5 = hidden addrspace(1) global ptr addrspace(1) null, align 8, !dbg.def ![[FILEVAR5_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR5_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR5_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR5_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(1))), argObjects: {![[FILEVAR5_F]]})
// CHECK-DAG: ![[FILEVAR5_F]] = distinct !DIFragment()
global int *global FileVar5;
// CHECK-DAG: @FileVar6 = hidden addrspace(1) global ptr addrspace(4) null, align 8, !dbg.def ![[FILEVAR6_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR6_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR6_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR6_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(4))), argObjects: {![[FILEVAR6_F]]})
// CHECK-DAG: ![[FILEVAR6_F]] = distinct !DIFragment()
constant int *global FileVar6;
// CHECK-DAG: @FileVar7 = hidden addrspace(1) global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4, !dbg.def ![[FILEVAR7_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR7_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar7", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR7_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR7_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(3))), argObjects: {![[FILEVAR7_F]]})
// CHECK-DAG: ![[FILEVAR7_F]] = distinct !DIFragment()
local int *global FileVar7;
// CHECK-DAG: @FileVar8 = hidden addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4, !dbg.def ![[FILEVAR8_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR8_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar8", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR8_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR8_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(5))), argObjects: {![[FILEVAR8_F]]})
// CHECK-DAG: ![[FILEVAR8_F]] = distinct !DIFragment()
private int *global FileVar8;
// CHECK-DAG: @FileVar9 = hidden addrspace(1) global ptr null, align 8, !dbg.def ![[FILEVAR9_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR9_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[FILEVAR9_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR9_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr)), argObjects: {![[FILEVAR9_F]]})
// CHECK-DAG: ![[FILEVAR9_F]] = distinct !DIFragment()
int *global FileVar9;

// CHECK-DAG: @FileVar10 = hidden addrspace(4) constant ptr addrspace(1) null, align 8, !dbg.def ![[FILEVAR10_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR10_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
// CHECK-DAG: ![[FILEVAR10_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR10_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr addrspace(1))), argObjects: {![[FILEVAR10_F]]})
// CHECK-DAG: ![[FILEVAR10_F]] = distinct !DIFragment()
global int *constant FileVar10 = 0;
// CHECK-DAG: @FileVar11 = hidden addrspace(4) constant ptr addrspace(4) null, align 8, !dbg.def ![[FILEVAR11_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR11_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
// CHECK-DAG: ![[FILEVAR11_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR11_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr addrspace(4))), argObjects: {![[FILEVAR11_F]]})
// CHECK-DAG: ![[FILEVAR11_F]] = distinct !DIFragment()
constant int *constant FileVar11 = 0;
// CHECK-DAG: @FileVar12 = hidden addrspace(4) constant ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4, !dbg.def ![[FILEVAR12_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR12_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar12", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
// CHECK-DAG: ![[FILEVAR12_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR12_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr addrspace(3))), argObjects: {![[FILEVAR12_F]]})
// CHECK-DAG: ![[FILEVAR12_F]] = distinct !DIFragment()
local int *constant FileVar12 = 0;
// CHECK-DAG: @FileVar13 = hidden addrspace(4) constant ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4, !dbg.def ![[FILEVAR13_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR13_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar13", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
// CHECK-DAG: ![[FILEVAR13_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR13_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr addrspace(5))), argObjects: {![[FILEVAR13_F]]})
// CHECK-DAG: ![[FILEVAR13_F]] = distinct !DIFragment()
private int *constant FileVar13 = 0;
// CHECK-DAG: @FileVar14 = hidden addrspace(4) constant ptr null, align 8, !dbg.def ![[FILEVAR14_F:[0-9]+]]
// CHECK-DAG: ![[FILEVAR14_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
// CHECK-DAG: ![[FILEVAR14_LT:[0-9]+]] = distinct !DILifetime(object: ![[FILEVAR14_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr)), argObjects: {![[FILEVAR14_F]]})
// CHECK-DAG: ![[FILEVAR14_F]] = distinct !DIFragment()
int *constant FileVar14 = 0;

kernel void kernel1(
    // CHECK-DAG: ![[KERNELARG0:[0-9]+]] = !DILocalVariable(name: "KernelArg0", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
    // CHECK-DAG: ![[KERNELARG0_LT:[0-9]+]] = distinct !DILifetime(object: ![[KERNELARG0]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(1))))
    // CHECK-DAG: call void @llvm.dbg.def(metadata ![[KERNELARG0_LT]], metadata ptr addrspace(5) %KernelArg0.addr), !dbg !{{[0-9]+}}
    global int *KernelArg0,
    // CHECK-DAG: ![[KERNELARG1:[0-9]+]] = !DILocalVariable(name: "KernelArg1", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
    // CHECK-DAG: ![[KERNELARG1_LT:[0-9]+]] = distinct !DILifetime(object: ![[KERNELARG1]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(4))))
    // CHECK-DAG: call void @llvm.dbg.def(metadata ![[KERNELARG1_LT]], metadata ptr addrspace(5) %KernelArg1.addr), !dbg !{{[0-9]+}}
    constant int *KernelArg1,
    // CHECK-DAG: ![[KERNELARG2:[0-9]+]] = !DILocalVariable(name: "KernelArg2", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
    // CHECK-DAG: ![[KERNELARG2_LT:[0-9]+]] = distinct !DILifetime(object: ![[KERNELARG2]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(3))))
    // CHECK-DAG: call void @llvm.dbg.def(metadata ![[KERNELARG2_LT]], metadata ptr addrspace(5) %KernelArg2.addr), !dbg !{{[0-9]+}}
    local int *KernelArg2) {
  private int *Tmp0;
  int *Tmp1;

  // CHECK-DAG: ![[FUNCVAR0:[0-9]+]] = !DILocalVariable(name: "FuncVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR0_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR0]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(1))))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR0_LT]], metadata ptr addrspace(5) %FuncVar0), !dbg !{{[0-9]+}}
  global int *FuncVar0 = KernelArg0;
  // CHECK-DAG: ![[FUNCVAR1:[0-9]+]] = !DILocalVariable(name: "FuncVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR1_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR1]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(4))))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR1_LT]], metadata ptr addrspace(5) %FuncVar1), !dbg !{{[0-9]+}}
  constant int *FuncVar1 = KernelArg1;
  // CHECK-DAG: ![[FUNCVAR2:[0-9]+]] = !DILocalVariable(name: "FuncVar2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR2_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR2]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(3))))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR2_LT]], metadata ptr addrspace(5) %FuncVar2), !dbg !{{[0-9]+}}
  local int *FuncVar2 = KernelArg2;
  // CHECK-DAG: ![[FUNCVAR3:[0-9]+]] = !DILocalVariable(name: "FuncVar3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR3_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR3]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(5))))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR3_LT]], metadata ptr addrspace(5) %FuncVar3), !dbg !{{[0-9]+}}
  private int *FuncVar3 = Tmp0;
  // CHECK-DAG: ![[FUNCVAR4:[0-9]+]] = !DILocalVariable(name: "FuncVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR4_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR4]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr)))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR4_LT]], metadata ptr addrspace(5) %FuncVar4), !dbg !{{[0-9]+}}
  int *FuncVar4 = Tmp1;

  // CHECK-DAG: ![[FUNCVAR5:[0-9]+]] = !DILocalVariable(name: "FuncVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR5_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR5]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(1))))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR5_LT]], metadata ptr addrspace(5) %FuncVar5), !dbg !{{[0-9]+}}
  global int *private FuncVar5 = KernelArg0;
  // CHECK-DAG: ![[FUNCVAR6:[0-9]+]] = !DILocalVariable(name: "FuncVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR6_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR6]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(4))))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR6_LT]], metadata ptr addrspace(5) %FuncVar6), !dbg !{{[0-9]+}}
  constant int *private FuncVar6 = KernelArg1;
  // CHECK-DAG: ![[FUNCVAR7:[0-9]+]] = !DILocalVariable(name: "FuncVar7", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR7_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR7]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(3))))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR7_LT]], metadata ptr addrspace(5) %FuncVar7), !dbg !{{[0-9]+}}
  local int *private FuncVar7 = KernelArg2;
  // CHECK-DAG: ![[FUNCVAR8:[0-9]+]] = !DILocalVariable(name: "FuncVar8", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR8_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR8]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(5))))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR8_LT]], metadata ptr addrspace(5) %FuncVar8), !dbg !{{[0-9]+}}
  private int *private FuncVar8 = Tmp0;
  // CHECK-DAG: ![[FUNCVAR9:[0-9]+]] = !DILocalVariable(name: "FuncVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_private)
  // CHECK-DAG: ![[FUNCVAR9_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR9]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr)))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR9_LT]], metadata ptr addrspace(5) %FuncVar9), !dbg !{{[0-9]+}}
  int *private FuncVar9 = Tmp1;

  // CHECK-DAG: @kernel1.FuncVar10 = internal addrspace(4) constant ptr addrspace(1) null, align 8, !dbg.def ![[FUNCVAR10_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR10_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
  // CHECK-DAG: ![[FUNCVAR10_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR10_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr addrspace(1))), argObjects: {![[FUNCVAR10_F]]})
  // CHECK-DAG: ![[FUNCVAR10_F]] = distinct !DIFragment()
  global int *constant FuncVar10 = 0;
  // CHECK-DAG: @kernel1.FuncVar11 = internal addrspace(4) constant ptr addrspace(4) null, align 8, !dbg.def ![[FUNCVAR11_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR11_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
  // CHECK-DAG: ![[FUNCVAR11_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR11_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr addrspace(4))), argObjects: {![[FUNCVAR11_F]]})
  // CHECK-DAG: ![[FUNCVAR11_F]] = distinct !DIFragment()
  constant int *constant FuncVar11 = 0;
  // CHECK-DAG: @kernel1.FuncVar12 = internal addrspace(4) constant ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), align 4, !dbg.def ![[FUNCVAR12_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR12_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar12", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
  // CHECK-DAG: ![[FUNCVAR12_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR12_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr addrspace(3))), argObjects: {![[FUNCVAR12_F]]})
  // CHECK-DAG: ![[FUNCVAR12_F]] = distinct !DIFragment()
  local int *constant FuncVar12 = 0;
  // CHECK-DAG: @kernel1.FuncVar13 = internal addrspace(4) constant ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4, !dbg.def ![[FUNCVAR13_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR13_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar13", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
  // CHECK-DAG: ![[FUNCVAR13_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR13_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr addrspace(5))), argObjects: {![[FUNCVAR13_F]]})
  // CHECK-DAG: ![[FUNCVAR13_F]] = distinct !DIFragment()
  private int *constant FuncVar13 = 0;
  // CHECK-DAG: @kernel1.FuncVar14 = internal addrspace(4) constant ptr null, align 8, !dbg.def ![[FUNCVAR14_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR14_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
  // CHECK-DAG: ![[FUNCVAR14_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR14_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(4)), DIOpDeref(ptr)), argObjects: {![[FUNCVAR14_F]]})
  // CHECK-DAG: ![[FUNCVAR14_F]] = distinct !DIFragment()
  int *constant FuncVar14 = 0;

  // CHECK-DAG: @kernel1.FuncVar15 = internal addrspace(3) global ptr addrspace(1) undef, align 8, !dbg.def ![[FUNCVAR15_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR15_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar15", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
  // CHECK-DAG: ![[FUNCVAR15_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR15_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(ptr addrspace(1))), argObjects: {![[FUNCVAR15_F]]})
  // CHECK-DAG: ![[FUNCVAR15_F]] = distinct !DIFragment()
  global int *local FuncVar15; FuncVar15 = KernelArg0;
  // CHECK-DAG: @kernel1.FuncVar16 = internal addrspace(3) global ptr addrspace(4) undef, align 8, !dbg.def ![[FUNCVAR16_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR16_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar16", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
  // CHECK-DAG: ![[FUNCVAR16_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR16_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(ptr addrspace(4))), argObjects: {![[FUNCVAR16_F]]})
  // CHECK-DAG: ![[FUNCVAR16_F]] = distinct !DIFragment()
  constant int *local FuncVar16; FuncVar16 = KernelArg1;
  // CHECK-DAG: @kernel1.FuncVar17 = internal addrspace(3) global ptr addrspace(3) undef, align 4, !dbg.def ![[FUNCVAR17_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR17_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar17", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
  // CHECK-DAG: ![[FUNCVAR17_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR17_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(ptr addrspace(3))), argObjects: {![[FUNCVAR17_F]]})
  // CHECK-DAG: ![[FUNCVAR17_F]] = distinct !DIFragment()
  local int *local FuncVar17; FuncVar17 = KernelArg2;
  // CHECK-DAG: @kernel1.FuncVar18 = internal addrspace(3) global ptr addrspace(5) undef, align 4, !dbg.def ![[FUNCVAR18_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR18_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar18", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
  // CHECK-DAG: ![[FUNCVAR18_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR18_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(ptr addrspace(5))), argObjects: {![[FUNCVAR18_F]]})
  // CHECK-DAG: ![[FUNCVAR18_F]] = distinct !DIFragment()
  private int *local FuncVar18; FuncVar18 = Tmp0;
  // CHECK-DAG: @kernel1.FuncVar19 = internal addrspace(3) global ptr undef, align 8, !dbg.def ![[FUNCVAR19_F:[0-9]+]]
  // CHECK-DAG: ![[FUNCVAR19_GV:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar19", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
  // CHECK-DAG: ![[FUNCVAR19_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR19_GV]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(ptr)), argObjects: {![[FUNCVAR19_F]]})
  // CHECK-DAG: ![[FUNCVAR19_F]] = distinct !DIFragment()
  int *local FuncVar19; FuncVar19 = Tmp1;
}

// FIXME: Can't figure out how to CHECK for this without hard-coding the arbitrary order of elements, and even in that case the fact that everything uses CHECK-DAG above seems to make it difficult to ensure this matches after all lifetime variables are defined: !llvm.dbg.retainedNodes = !{![[FILEVAR0_LT]], ![[FILEVAR1_LT]], ![[FILEVAR2_LT]], ![[FILEVAR3_LT]], ![[FILEVAR4_LT]], ![[FILEVAR5_LT]], ![[FILEVAR6_LT]], ![[FILEVAR7_LT]], ![[FILEVAR8_LT]], ![[FILEVAR9_LT]], !![[FUNCVAR10_LT]], ![[FUNCVAR11_LT]], ![[FUNCVAR12_LT]], ![[FUNCVAR13_LT]], ![[FUNCVAR14_LT]], ![[FUNCVAR15_LT]], ![[FUNCVAR16_LT]], ![[FUNCVAR17_LT]], ![[FUNCVAR18_LT]], ![[FUNCVAR19_LT]]}
