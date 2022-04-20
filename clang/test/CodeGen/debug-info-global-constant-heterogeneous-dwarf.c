// RUN: %clang_cc1 -D ARG_TYPE=int -D PTR_ARG='&g' -D VAL_ARG=g -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=INT-ADDROF-VAL %s
// RUN: %clang_cc1 -D ARG_TYPE=int -D PTR_ARG='&g' -D VAL_ARG=0 -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=INT-ADDROF-NOVAL %s
// RUN: %clang_cc1 -D ARG_TYPE=int -D PTR_ARG=0 -D VAL_ARG=g -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=INT-NOADDROF-VAL %s
// RUN: %clang_cc1 -D ARG_TYPE=int -D PTR_ARG=0 -D VAL_ARG=0 -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=INT-NOADDROF-NOVAL %s
//
// RUN: %clang_cc1 -D ARG_TYPE=float -D PTR_ARG='&g' -D VAL_ARG=g -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=FLOAT-ADDROF-VAL %s
// RUN: %clang_cc1 -D ARG_TYPE=float -D PTR_ARG='&g' -D VAL_ARG=0 -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=FLOAT-ADDROF-NOVAL %s
// RUN: %clang_cc1 -D ARG_TYPE=float -D PTR_ARG=0 -D VAL_ARG=g -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=FLOAT-NOADDROF-VAL %s
// RUN: %clang_cc1 -D ARG_TYPE=float -D PTR_ARG=0 -D VAL_ARG=0 -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=FLOAT-NOADDROF-NOVAL %s

// INT-ADDROF-VAL: @g = internal constant i32 1, align 4{{$}}
// INT-ADDROF-VAL-DAG: !llvm.dbg.retainedNodes = !{![[#LIFETIME:]]}
// INT-ADDROF-VAL-DAG: ![[#LIFETIME]] = distinct !DILifetime(object: ![[#GV:]], location: !DIExpr(DIOpConstant(i32 1)))
// INT-ADDROF-VAL-DAG: ![[#GV]] = distinct !DIGlobalVariable(name: "g",
//
// INT-ADDROF-NOVAL: @g = internal constant i32 1, align 4, !dbg.def ![[#FRAGMENT:]]
// INT-ADDROF-NOVAL-DAG: !llvm.dbg.retainedNodes = !{![[#LIFETIME:]]}
// INT-ADDROF-NOVAL-DAG: ![[#FRAGMENT]] = distinct !DIFragment()
// INT-ADDROF-NOVAL-DAG: ![[#LIFETIME]] = distinct !DILifetime(object: ![[#GV:]], location: !DIExpr(DIOpArg(0, ptr), DIOpDeref(i32)), argObjects: {![[#FRAGMENT]]})
// INT-ADDROF-NOVAL-DAG: ![[#GV]] = distinct !DIGlobalVariable(name: "g",
//
// INT-NOADDROF-VAL-NOT: @g =
// INT-NOADDROF-VAL-DAG: !llvm.dbg.retainedNodes = !{![[#LIFETIME:]]}
// INT-NOADDROF-VAL-DAG: ![[#LIFETIME]] = distinct !DILifetime(object: ![[#GV:]], location: !DIExpr(DIOpConstant(i32 1)))
// INT-NOADDROF-VAL-DAG: ![[#GV]] = distinct !DIGlobalVariable(name: "g",
//
// INT-NOADDROF-NOVAL-NOT: @g =
// INT-NOADDROF-NOVAL-NOT: !llvm.dbg.retainedNodes =

// FLOAT-ADDROF-VAL: @g = internal constant float 1.000000e+00, align 4{{$}}
// FLOAT-ADDROF-VAL-DAG: !llvm.dbg.retainedNodes = !{![[#LIFETIME:]]}
// FLOAT-ADDROF-VAL-DAG: ![[#LIFETIME]] = distinct !DILifetime(object: ![[#GV:]], location: !DIExpr(DIOpConstant(float 1.000000e+00)))
// FLOAT-ADDROF-VAL-DAG: ![[#GV]] = distinct !DIGlobalVariable(name: "g",
//
// FLOAT-ADDROF-NOVAL: @g = internal constant float 1.000000e+00, align 4, !dbg.def ![[#FRAGMENT:]]
// FLOAT-ADDROF-NOVAL-DAG: !llvm.dbg.retainedNodes = !{![[#LIFETIME:]]}
// FLOAT-ADDROF-NOVAL-DAG: ![[#FRAGMENT]] = distinct !DIFragment()
// FLOAT-ADDROF-NOVAL-DAG: ![[#LIFETIME]] = distinct !DILifetime(object: ![[#GV:]], location: !DIExpr(DIOpArg(0, ptr), DIOpDeref(float)), argObjects: {![[#FRAGMENT]]})
// FLOAT-ADDROF-NOVAL-DAG: ![[#GV]] = distinct !DIGlobalVariable(name: "g",
//
// FLOAT-NOADDROF-VAL-NOT: @g =
// FLOAT-NOADDROF-VAL-DAG: !llvm.dbg.retainedNodes = !{![[#LIFETIME:]]}
// FLOAT-NOADDROF-VAL-DAG: ![[#LIFETIME]] = distinct !DILifetime(object: ![[#GV:]], location: !DIExpr(DIOpConstant(float 1.000000e+00)))
// FLOAT-NOADDROF-VAL-DAG: ![[#GV]] = distinct !DIGlobalVariable(name: "g",
//
// FLOAT-NOADDROF-NOVAL-NOT: @g =
// FLOAT-NOADDROF-NOVAL-NOT: !llvm.dbg.retainedNodes =

static const ARG_TYPE g = 1;
void callee(const ARG_TYPE *, ARG_TYPE);
void caller() {
  callee(PTR_ARG, VAL_ARG);
}
