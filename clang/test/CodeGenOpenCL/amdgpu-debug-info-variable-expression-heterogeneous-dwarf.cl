// REQUIRES: amdgpu-registered-target
// RUN: %clang -cl-std=CL2.0 -emit-llvm -g -gheterogeneous-dwarf -O0 -S -nogpulib -target amdgcn-amd-amdhsa -mcpu=fiji -o - %s | FileCheck %s
// RUN: %clang -cl-std=CL2.0 -emit-llvm -g -gheterogeneous-dwarf -O0 -S -nogpulib -target amdgcn-amd-amdhsa-opencl -mcpu=fiji -o - %s | FileCheck %s

kernel void kernel1(
    // CHECK-DAG: ![[KERNELARG0:[0-9]+]] = !DILocalVariable(name: "KernelArg0", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
    // CHECK-DAG: ![[KERNELARG0_LT:[0-9]+]] = distinct !DILifetime(object: ![[KERNELARG0]], location: !DIExpr(DIOpReferrer(i32 addrspace(1)* addrspace(5)*), DIOpDeref()))
    // CHECK-DAG: call void @llvm.dbg.def(metadata ![[KERNELARG0_LT]], metadata i32 addrspace(1)* addrspace(5)* %KernelArg0.addr), !dbg !{{[0-9]+}}
    global int *KernelArg0,
    // CHECK-DAG: ![[KERNELARG1:[0-9]+]] = !DILocalVariable(name: "KernelArg1", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
    // CHECK-DAG: ![[KERNELARG1_LT:[0-9]+]] = distinct !DILifetime(object: ![[KERNELARG1]], location: !DIExpr(DIOpReferrer(i32 addrspace(4)* addrspace(5)*), DIOpDeref()))
    // CHECK-DAG: call void @llvm.dbg.def(metadata ![[KERNELARG1_LT]], metadata i32 addrspace(4)* addrspace(5)* %KernelArg1.addr), !dbg !{{[0-9]+}}
    constant int *KernelArg1,
    // CHECK-DAG: ![[KERNELARG2:[0-9]+]] = !DILocalVariable(name: "KernelArg2", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
    // CHECK-DAG: ![[KERNELARG2_LT:[0-9]+]] = distinct !DILifetime(object: ![[KERNELARG2]], location: !DIExpr(DIOpReferrer(i32 addrspace(3)* addrspace(5)*), DIOpDeref()))
    // CHECK-DAG: call void @llvm.dbg.def(metadata ![[KERNELARG2_LT]], metadata i32 addrspace(3)* addrspace(5)* %KernelArg2.addr), !dbg !{{[0-9]+}}
    local int *KernelArg2) {
  private int *Tmp0;
  int *Tmp1;

  // CHECK-DAG: ![[FUNCVAR0:[0-9]+]] = !DILocalVariable(name: "FuncVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR0_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR0]], location: !DIExpr(DIOpReferrer(i32 addrspace(1)* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR0_LT]], metadata i32 addrspace(1)* addrspace(5)* %FuncVar0), !dbg !{{[0-9]+}}
  global int *FuncVar0 = KernelArg0;
  // CHECK-DAG: ![[FUNCVAR1:[0-9]+]] = !DILocalVariable(name: "FuncVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR1_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR1]], location: !DIExpr(DIOpReferrer(i32 addrspace(4)* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR1_LT]], metadata i32 addrspace(4)* addrspace(5)* %FuncVar1), !dbg !{{[0-9]+}}
  constant int *FuncVar1 = KernelArg1;
  // CHECK-DAG: ![[FUNCVAR2:[0-9]+]] = !DILocalVariable(name: "FuncVar2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR2_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR2]], location: !DIExpr(DIOpReferrer(i32 addrspace(3)* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR2_LT]], metadata i32 addrspace(3)* addrspace(5)* %FuncVar2), !dbg !{{[0-9]+}}
  local int *FuncVar2 = KernelArg2;
  // CHECK-DAG: ![[FUNCVAR3:[0-9]+]] = !DILocalVariable(name: "FuncVar3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR3_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR3]], location: !DIExpr(DIOpReferrer(i32 addrspace(5)* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR3_LT]], metadata i32 addrspace(5)* addrspace(5)* %FuncVar3), !dbg !{{[0-9]+}}
  private int *FuncVar3 = Tmp0;
  // CHECK-DAG: ![[FUNCVAR4:[0-9]+]] = !DILocalVariable(name: "FuncVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR4_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR4]], location: !DIExpr(DIOpReferrer(i32* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR4_LT]], metadata i32* addrspace(5)* %FuncVar4), !dbg !{{[0-9]+}}
  int *FuncVar4 = Tmp1;

  // CHECK-DAG: ![[FUNCVAR5:[0-9]+]] = !DILocalVariable(name: "FuncVar5", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR5_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR5]], location: !DIExpr(DIOpReferrer(i32 addrspace(1)* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR5_LT]], metadata i32 addrspace(1)* addrspace(5)* %FuncVar5), !dbg !{{[0-9]+}}
  global int *private FuncVar5 = KernelArg0;
  // CHECK-DAG: ![[FUNCVAR6:[0-9]+]] = !DILocalVariable(name: "FuncVar6", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR6_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR6]], location: !DIExpr(DIOpReferrer(i32 addrspace(4)* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR6_LT]], metadata i32 addrspace(4)* addrspace(5)* %FuncVar6), !dbg !{{[0-9]+}}
  constant int *private FuncVar6 = KernelArg1;
  // CHECK-DAG: ![[FUNCVAR7:[0-9]+]] = !DILocalVariable(name: "FuncVar7", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR7_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR7]], location: !DIExpr(DIOpReferrer(i32 addrspace(3)* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR7_LT]], metadata i32 addrspace(3)* addrspace(5)* %FuncVar7), !dbg !{{[0-9]+}}
  local int *private FuncVar7 = KernelArg2;
  // CHECK-DAG: ![[FUNCVAR8:[0-9]+]] = !DILocalVariable(name: "FuncVar8", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR8_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR8]], location: !DIExpr(DIOpReferrer(i32 addrspace(5)* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR8_LT]], metadata i32 addrspace(5)* addrspace(5)* %FuncVar8), !dbg !{{[0-9]+}}
  private int *private FuncVar8 = Tmp0;
  // CHECK-DAG: ![[FUNCVAR9:[0-9]+]] = !DILocalVariable(name: "FuncVar9", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: ![[FUNCVAR9_LT:[0-9]+]] = distinct !DILifetime(object: ![[FUNCVAR9]], location: !DIExpr(DIOpReferrer(i32* addrspace(5)*), DIOpDeref()))
  // CHECK-DAG: call void @llvm.dbg.def(metadata ![[FUNCVAR9_LT]], metadata i32* addrspace(5)* %FuncVar9), !dbg !{{[0-9]+}}
  int *private FuncVar9 = Tmp1;
}
