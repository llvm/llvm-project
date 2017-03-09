// RUN: %clang -cl-std=CL2.0 -emit-llvm -g -O0 -S -target amdgcn-amd-amdhsa -mcpu=fiji -o - %s | FileCheck %s

// CHECK-DAG: ![[NONE:[0-9]+]] = !DIExpression()
// CHECK-DAG: ![[LOCAL:[0-9]+]] = !DIExpression(DW_OP_constu, 2, DW_OP_swap, DW_OP_xderef)
// CHECK-DAG: ![[PRIVATE:[0-9]+]] = !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)

// CHECK-DAG: ![[FILEVAR0:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR0]])
global int *FileVar0;
// CHECK-DAG: ![[FILEVAR1:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR1]])
constant int *FileVar1;
// CHECK-DAG: ![[FILEVAR2:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR2]])
local int *FileVar2;
// CHECK-DAG: ![[FILEVAR3:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR3]])
private int *FileVar3;
// CHECK-DAG: ![[FILEVAR4:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR4]])
int *FileVar4;

// CHECK-DAG: ![[FILEVAR5:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR5]])
global int *global FileVar5;
// CHECK-DAG: ![[FILEVAR6:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR6]])
constant int *global FileVar6;
// CHECK-DAG: ![[FILEVAR7:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar7", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR7]])
local int *global FileVar7;
// CHECK-DAG: ![[FILEVAR8:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar8", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR8]])
private int *global FileVar8;
// CHECK-DAG: ![[FILEVAR9:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR9]])
int *global FileVar9;

// CHECK-DAG: ![[FILEVAR10:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR10]])
global int *constant FileVar10 = 0;
// CHECK-DAG: ![[FILEVAR11:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR11]])
constant int *constant FileVar11 = 0;
// CHECK-DAG: ![[FILEVAR12:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar12", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR12]])
local int *constant FileVar12 = 0;
// CHECK-DAG: ![[FILEVAR13:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar13", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR13]])
private int *constant FileVar13 = 0;
// CHECK-DAG: ![[FILEVAR14:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR14]])
int *constant FileVar14 = 0;

kernel void kernel1(
    // CHECK-DAG: ![[KERNELARG0:[0-9]+]] = !DILocalVariable(name: "KernelArg0", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
    // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(1)** {{.*}}, metadata ![[KERNELARG0]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
    global int *KernelArg0,
    // CHECK-DAG: ![[KERNELARG1:[0-9]+]] = !DILocalVariable(name: "KernelArg1", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
    // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(2)** {{.*}}, metadata ![[KERNELARG1]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
    constant int *KernelArg1,
    // CHECK-DAG: ![[KERNELARG2:[0-9]+]] = !DILocalVariable(name: "KernelArg2", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
    // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(3)** {{.*}}, metadata ![[KERNELARG2]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
    local int *KernelArg2) {
  private int *Tmp0;
  int *Tmp1;

  // CHECK-DAG: ![[FUNCVAR0:[0-9]+]] = !DILocalVariable(name: "FuncVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(1)** {{.*}}, metadata ![[FUNCVAR0]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  global int *FuncVar0 = KernelArg0;
  // CHECK-DAG: ![[FUNCVAR1:[0-9]+]] = !DILocalVariable(name: "FuncVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(2)** {{.*}}, metadata ![[FUNCVAR1]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  constant int *FuncVar1 = KernelArg1;
  // CHECK-DAG: ![[FUNCVAR2:[0-9]+]] = !DILocalVariable(name: "FuncVar2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(3)** {{.*}}, metadata ![[FUNCVAR2]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  local int *FuncVar2 = KernelArg2;
  // CHECK-DAG: ![[FUNCVAR3:[0-9]+]] = !DILocalVariable(name: "FuncVar3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32** {{.*}}, metadata ![[FUNCVAR3]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  private int *FuncVar3 = Tmp0;
  // CHECK-DAG: ![[FUNCVAR4:[0-9]+]] = !DILocalVariable(name: "FuncVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(4)** {{.*}}, metadata ![[FUNCVAR4]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  int *FuncVar4 = Tmp1;

  // CHECK-DAG: ![[FUNCVAR5:[0-9]+]] = !DILocalVariable(name: "FuncVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(1)** {{.*}}, metadata ![[FUNCVAR5]], metadata ![[NONE]]), !dbg !{{[0-9]+}}
  global int *constant FuncVar5 = KernelArg0;
  // CHECK-DAG: ![[FUNCVAR6:[0-9]+]] = !DILocalVariable(name: "FuncVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(2)** {{.*}}, metadata ![[FUNCVAR6]], metadata ![[NONE]]), !dbg !{{[0-9]+}}
  constant int *constant FuncVar6 = KernelArg1;
  // CHECK-DAG: ![[FUNCVAR7:[0-9]+]] = !DILocalVariable(name: "FuncVar7", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(3)** {{.*}}, metadata ![[FUNCVAR7]], metadata ![[NONE]]), !dbg !{{[0-9]+}}
  local int *constant FuncVar7 = KernelArg2;
  // CHECK-DAG: ![[FUNCVAR8:[0-9]+]] = !DILocalVariable(name: "FuncVar8", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32** {{.*}}, metadata ![[FUNCVAR8]], metadata ![[NONE]]), !dbg !{{[0-9]+}}
  private int *constant FuncVar8 = Tmp0;
  // CHECK-DAG: ![[FUNCVAR9:[0-9]+]] = !DILocalVariable(name: "FuncVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(4)** {{.*}}, metadata ![[FUNCVAR9]], metadata ![[NONE]]), !dbg !{{[0-9]+}}
  int *constant FuncVar9 = Tmp1;

  // CHECK-DAG: ![[FUNCVAR10:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar10", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true)
  // CHECK-DAG: !DIGlobalVariableExpression(var: ![[FUNCVAR10]], expr: ![[LOCAL]])
  global int *local FuncVar10; FuncVar10 = KernelArg0;
  // CHECK-DAG: ![[FUNCVAR11:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar11", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true)
  // CHECK-DAG: !DIGlobalVariableExpression(var: ![[FUNCVAR11]], expr: ![[LOCAL]])
  constant int *local FuncVar11; FuncVar11 = KernelArg1;
  // CHECK-DAG: ![[FUNCVAR12:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar12", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true)
  // CHECK-DAG: !DIGlobalVariableExpression(var: ![[FUNCVAR12]], expr: ![[LOCAL]])
  local int *local FuncVar12; FuncVar12 = KernelArg2;
  // CHECK-DAG: ![[FUNCVAR13:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar13", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true)
  // CHECK-DAG: !DIGlobalVariableExpression(var: ![[FUNCVAR13]], expr: ![[LOCAL]])
  private int *local FuncVar13; FuncVar13 = Tmp0;
  // CHECK-DAG: ![[FUNCVAR14:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar14", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true)
  // CHECK-DAG: !DIGlobalVariableExpression(var: ![[FUNCVAR14]], expr: ![[LOCAL]])
  int *local FuncVar14; FuncVar14 = Tmp1;

  // CHECK-DAG: ![[FUNCVAR15:[0-9]+]] = !DILocalVariable(name: "FuncVar15", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(1)** {{.*}}, metadata ![[FUNCVAR15]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  global int *private FuncVar15 = KernelArg0;
  // CHECK-DAG: ![[FUNCVAR16:[0-9]+]] = !DILocalVariable(name: "FuncVar16", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(2)** {{.*}}, metadata ![[FUNCVAR16]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  constant int *private FuncVar16 = KernelArg1;
  // CHECK-DAG: ![[FUNCVAR17:[0-9]+]] = !DILocalVariable(name: "FuncVar17", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(3)** {{.*}}, metadata ![[FUNCVAR17]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  local int *private FuncVar17 = KernelArg2;
  // CHECK-DAG: ![[FUNCVAR18:[0-9]+]] = !DILocalVariable(name: "FuncVar18", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32** {{.*}}, metadata ![[FUNCVAR18]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  private int *private FuncVar18 = Tmp0;
  // CHECK-DAG: ![[FUNCVAR19:[0-9]+]] = !DILocalVariable(name: "FuncVar19", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata i32 addrspace(4)** {{.*}}, metadata ![[FUNCVAR19]], metadata ![[PRIVATE]]), !dbg !{{[0-9]+}}
  int *private FuncVar19 = Tmp1;
}
