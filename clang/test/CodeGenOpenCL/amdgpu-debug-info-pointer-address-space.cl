// REQUIRES: amdgpu-registered-target
// RUN: %clang -cl-std=CL2.0 -emit-llvm -g -O0 -S -nogpulib -target amdgcn-amd-amdhsa -mcpu=fiji -o - %s | FileCheck %s
// RUN: %clang -cl-std=CL2.0 -emit-llvm -g -O0 -S -nogpulib -target amdgcn-amd-amdhsa-opencl -mcpu=fiji -o - %s | FileCheck %s

<<<<<<< HEAD
// CHECK-DAG: ![[DWARF_MEMORY_SPACE_GLOBAL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: ![[DWARF_MEMORY_SPACE_CONSTANT:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, memorySpace: DW_MSPACE_LLVM_constant)
// CHECK-DAG: ![[DWARF_MEMORY_SPACE_LOCAL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, addressSpace: 3, memorySpace: DW_MSPACE_LLVM_group)
// CHECK-DAG: ![[DWARF_MEMORY_SPACE_PRIVATE:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, addressSpace: 5, memorySpace: DW_MSPACE_LLVM_private)
// CHECK-DAG: ![[DWARF_MEMORY_SPACE_NONE:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, addressSpace: 1)
||||||| 1287ed1fa2db
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_NONE:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}})
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_LOCAL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 2)
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_PRIVATE:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 1)
=======
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_GLOBAL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}})
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_LOCAL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 3)
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_PRIVATE:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 5)
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_GENERIC:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 1)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e

<<<<<<< HEAD
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_GLOBAL]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
||||||| 1287ed1fa2db
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: false, isDefinition: true)
=======
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: false, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
global int *FileVar0;
<<<<<<< HEAD
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_CONSTANT]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
||||||| 1287ed1fa2db
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: false, isDefinition: true)
=======
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: false, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
constant int *FileVar1;
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_LOCAL]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
local int *FileVar2;
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_PRIVATE]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
private int *FileVar3;
<<<<<<< HEAD
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_NONE]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
||||||| 1287ed1fa2db
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: false, isDefinition: true)
=======
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GENERIC]], isLocal: false, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
int *FileVar4;

<<<<<<< HEAD
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_GLOBAL]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
||||||| 1287ed1fa2db
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: false, isDefinition: true)
=======
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: false, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
global int *global FileVar5;
<<<<<<< HEAD
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_CONSTANT]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
||||||| 1287ed1fa2db
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: false, isDefinition: true)
=======
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: false, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
constant int *global FileVar6;
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar7", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_LOCAL]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
local int *global FileVar7;
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar8", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_PRIVATE]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
private int *global FileVar8;
<<<<<<< HEAD
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_NONE]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
||||||| 1287ed1fa2db
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: false, isDefinition: true)
=======
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GENERIC]], isLocal: false, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
int *global FileVar9;

<<<<<<< HEAD
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_GLOBAL]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
||||||| 1287ed1fa2db
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: false, isDefinition: true)
=======
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: false, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
global int *constant FileVar10 = 0;
<<<<<<< HEAD
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_CONSTANT]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
||||||| 1287ed1fa2db
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: false, isDefinition: true)
=======
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: false, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
constant int *constant FileVar11 = 0;
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar12", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_LOCAL]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
local int *constant FileVar12 = 0;
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar13", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_PRIVATE]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
private int *constant FileVar13 = 0;
<<<<<<< HEAD
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_NONE]], isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
||||||| 1287ed1fa2db
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: false, isDefinition: true)
=======
// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GENERIC]], isLocal: false, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
int *constant FileVar14 = 0;

kernel void kernel1(
<<<<<<< HEAD
    // CHECK-DAG: !DILocalVariable(name: "KernelArg0", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_GLOBAL]], memorySpace: DW_MSPACE_LLVM_private)
||||||| 1287ed1fa2db
    // CHECK-DAG: !DILocalVariable(name: "KernelArg0", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]])
=======
    // CHECK-DAG: !DILocalVariable(name: "KernelArg0", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]])
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
    global int *KernelArg0,
<<<<<<< HEAD
    // CHECK-DAG: !DILocalVariable(name: "KernelArg1", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_CONSTANT]], memorySpace: DW_MSPACE_LLVM_private)
||||||| 1287ed1fa2db
    // CHECK-DAG: !DILocalVariable(name: "KernelArg1", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]])
=======
    // CHECK-DAG: !DILocalVariable(name: "KernelArg1", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]])
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
    constant int *KernelArg1,
    // CHECK-DAG: !DILocalVariable(name: "KernelArg2", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_LOCAL]], memorySpace: DW_MSPACE_LLVM_private)
    local int *KernelArg2) {
  private int *Tmp0;
  int *Tmp1;

<<<<<<< HEAD
  // CHECK-DAG: !DILocalVariable(name: "FuncVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_GLOBAL]], memorySpace: DW_MSPACE_LLVM_private)
||||||| 1287ed1fa2db
  // CHECK-DAG: !DILocalVariable(name: "FuncVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]])
=======
  // CHECK-DAG: !DILocalVariable(name: "FuncVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]])
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  global int *FuncVar0 = KernelArg0;
<<<<<<< HEAD
  // CHECK-DAG: !DILocalVariable(name: "FuncVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_CONSTANT]], memorySpace: DW_MSPACE_LLVM_private)
||||||| 1287ed1fa2db
  // CHECK-DAG: !DILocalVariable(name: "FuncVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]])
=======
  // CHECK-DAG: !DILocalVariable(name: "FuncVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]])
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  constant int *FuncVar1 = KernelArg1;
  // CHECK-DAG: !DILocalVariable(name: "FuncVar2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_LOCAL]], memorySpace: DW_MSPACE_LLVM_private)
  local int *FuncVar2 = KernelArg2;
  // CHECK-DAG: !DILocalVariable(name: "FuncVar3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_PRIVATE]], memorySpace: DW_MSPACE_LLVM_private)
  private int *FuncVar3 = Tmp0;
<<<<<<< HEAD
  // CHECK-DAG: !DILocalVariable(name: "FuncVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_NONE]], memorySpace: DW_MSPACE_LLVM_private)
||||||| 1287ed1fa2db
  // CHECK-DAG: !DILocalVariable(name: "FuncVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]])
=======
  // CHECK-DAG: !DILocalVariable(name: "FuncVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GENERIC]])
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  int *FuncVar4 = Tmp1;

<<<<<<< HEAD
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_GLOBAL]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
||||||| 1287ed1fa2db
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: true, isDefinition: true)
=======
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: true, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  global int *constant FuncVar5 = 0;
<<<<<<< HEAD
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_CONSTANT]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
||||||| 1287ed1fa2db
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: true, isDefinition: true)
=======
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: true, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  constant int *constant FuncVar6 = 0;
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar7", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_LOCAL]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
  local int *constant FuncVar7 = 0;
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar8", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_PRIVATE]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
  private int *constant FuncVar8 = 0;
<<<<<<< HEAD
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_NONE]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
||||||| 1287ed1fa2db
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: true, isDefinition: true)
=======
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GENERIC]], isLocal: true, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  int *constant FuncVar9 = 0;

<<<<<<< HEAD
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_GLOBAL]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
||||||| 1287ed1fa2db
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: true, isDefinition: true)
=======
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: true, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  global int *local FuncVar10; FuncVar10 = KernelArg0;
<<<<<<< HEAD
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_CONSTANT]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
||||||| 1287ed1fa2db
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: true, isDefinition: true)
=======
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: true, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  constant int *local FuncVar11; FuncVar11 = KernelArg1;
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar12", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_LOCAL]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
  local int *local FuncVar12; FuncVar12 = KernelArg2;
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar13", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_PRIVATE]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
  private int *local FuncVar13; FuncVar13 = Tmp0;
<<<<<<< HEAD
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_NONE]], isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
||||||| 1287ed1fa2db
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]], isLocal: true, isDefinition: true)
=======
  // CHECK-DAG: distinct !DIGlobalVariable(name: "FuncVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GENERIC]], isLocal: true, isDefinition: true)
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  int *local FuncVar14; FuncVar14 = Tmp1;

<<<<<<< HEAD
  // CHECK-DAG: !DILocalVariable(name: "FuncVar15", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_GLOBAL]], memorySpace: DW_MSPACE_LLVM_private)
||||||| 1287ed1fa2db
  // CHECK-DAG: !DILocalVariable(name: "FuncVar15", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]])
=======
  // CHECK-DAG: !DILocalVariable(name: "FuncVar15", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]])
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  global int *private FuncVar15 = KernelArg0;
<<<<<<< HEAD
  // CHECK-DAG: !DILocalVariable(name: "FuncVar16", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_CONSTANT]], memorySpace: DW_MSPACE_LLVM_private)
||||||| 1287ed1fa2db
  // CHECK-DAG: !DILocalVariable(name: "FuncVar16", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]])
=======
  // CHECK-DAG: !DILocalVariable(name: "FuncVar16", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]])
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  constant int *private FuncVar16 = KernelArg1;
  // CHECK-DAG: !DILocalVariable(name: "FuncVar17", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_LOCAL]], memorySpace: DW_MSPACE_LLVM_private)
  local int *private FuncVar17 = KernelArg2;
  // CHECK-DAG: !DILocalVariable(name: "FuncVar18", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_PRIVATE]], memorySpace: DW_MSPACE_LLVM_private)
  private int *private FuncVar18 = Tmp0;
<<<<<<< HEAD
  // CHECK-DAG: !DILocalVariable(name: "FuncVar19", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_MEMORY_SPACE_NONE]], memorySpace: DW_MSPACE_LLVM_private)
||||||| 1287ed1fa2db
  // CHECK-DAG: !DILocalVariable(name: "FuncVar19", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_NONE]])
=======
  // CHECK-DAG: !DILocalVariable(name: "FuncVar19", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GENERIC]])
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  int *private FuncVar19 = Tmp1;
}

struct FileStruct0 {
<<<<<<< HEAD
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_GLOBAL]], size: {{[0-9]+}})
||||||| 1287ed1fa2db
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_NONE]], size: {{[0-9]+}})
=======
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_GLOBAL]], size: {{[0-9]+}})
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  global int *StructMem0;
<<<<<<< HEAD
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_CONSTANT]], size: {{[0-9]+}}, offset: {{[0-9]+}})
||||||| 1287ed1fa2db
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_NONE]], size: {{[0-9]+}}, offset: {{[0-9]+}})
=======
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_GLOBAL]], size: {{[0-9]+}}, offset: {{[0-9]+}})
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  constant int *StructMem1;
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_LOCAL]], size: {{[0-9]+}}, offset: {{[0-9]+}})
  local int *StructMem2;
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_PRIVATE]], size: {{[0-9]+}}, offset: {{[0-9]+}})
  private int *StructMem3;
<<<<<<< HEAD
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_NONE]], size: {{[0-9]+}}, offset: {{[0-9]+}})
||||||| 1287ed1fa2db
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_NONE]], size: {{[0-9]+}}, offset: {{[0-9]+}})
=======
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "StructMem4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_GENERIC]], size: {{[0-9]+}}, offset: {{[0-9]+}})
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
  int *StructMem4;
};

struct FileStruct1 {
  union {
<<<<<<< HEAD
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_GLOBAL]], size: {{[0-9]+}})
||||||| 1287ed1fa2db
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_NONE]], size: {{[0-9]+}})
=======
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_GLOBAL]], size: {{[0-9]+}})
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
    global int *UnionMem0;
<<<<<<< HEAD
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_CONSTANT]], size: {{[0-9]+}})
||||||| 1287ed1fa2db
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_NONE]], size: {{[0-9]+}})
=======
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_GLOBAL]], size: {{[0-9]+}})
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
    constant int *UnionMem1;
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_LOCAL]], size: {{[0-9]+}})
    local int *UnionMem2;
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_PRIVATE]], size: {{[0-9]+}})
    private int *UnionMem3;
<<<<<<< HEAD
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_MEMORY_SPACE_NONE]], size: {{[0-9]+}})
||||||| 1287ed1fa2db
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_NONE]], size: {{[0-9]+}})
=======
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "UnionMem4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: ![[DWARF_ADDRESS_SPACE_GENERIC]], size: {{[0-9]+}})
>>>>>>> 4a8bb08bd2261afe22c8addea9f830906084869e
    int *UnionMem4;
  };
  long StructMem0;
};

kernel void kernel2(global struct FileStruct0 *Kernel2Arg0,
                    global struct FileStruct1 *Kernel2Arg1) {}
