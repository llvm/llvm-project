// RUN: %clang_cc1 -D PTR_ARG='&i' -D INT_ARG=i -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=ADDROF-VAL %s
// RUN: %clang_cc1 -D PTR_ARG='&i' -D INT_ARG=0 -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=ADDROF-NOVAL %s
// RUN: %clang_cc1 -D PTR_ARG=0 -D INT_ARG=i -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=NOADDROF-VAL %s
// RUN: %clang_cc1 -D PTR_ARG=0 -D INT_ARG=0 -emit-llvm -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck --check-prefix=NOADDROF-NOVAL %s

// ADDROF-VAL: @i = internal constant i32 1, align 4, !dbg.def ![[#FRAGMENT:]]
// ADDROF-NOVAL: @i = internal constant i32 1, align 4, !dbg.def ![[#FRAGMENT:]]
// NOADDROF-VAL-NOT: @i =
// NOADDROF-NOVAL-NOT: @i =
static const int i = 1;

// ADDROF-VAL-DAG: !llvm.dbg.retainedNodes = !{![[#LIFETIME:]]}
// ADDROF-VAL-DAG: ![[#FRAGMENT]] = distinct !DIFragment()
// ADDROF-VAL-DAG: ![[#LIFETIME]] = distinct !DILifetime(object: ![[#GV:]], location: !DIExpr(DIOpArg(0, i32*), DIOpDeref(i32)), argObjects: {![[#FRAGMENT]]})
// ADDROF-VAL-DAG: ![[#GV]] = distinct !DIGlobalVariable(name: "i",

// ADDROF-NOVAL-DAG: !llvm.dbg.retainedNodes = !{![[#LIFETIME:]]}
// ADDROF-NOVAL-DAG: ![[#FRAGMENT]] = distinct !DIFragment()
// ADDROF-NOVAL-DAG: ![[#LIFETIME]] = distinct !DILifetime(object: ![[#GV:]], location: !DIExpr(DIOpArg(0, i32*), DIOpDeref(i32)), argObjects: {![[#FRAGMENT]]})
// ADDROF-NOVAL-DAG: ![[#GV]] = distinct !DIGlobalVariable(name: "i",

// NOADDROF-VAL-NOT: !llvm.dbg.retainedNodes =

// NOADDROF-NOVAL-NOT: !llvm.dbg.retainedNodes =

void g(const int *, int);
void f() {
  g(PTR_ARG, INT_ARG);
}
