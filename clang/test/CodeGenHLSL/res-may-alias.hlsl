// RUN: %clang_dxc -res-may-alias -T lib_6_3 -HV 202x -Vd -Xclang -emit-llvm %s | FileCheck %s --check-prefix=FLAG
// RUN: %clang_dxc -T lib_6_3 -HV 202x -Vd -Xclang -emit-llvm %s | FileCheck %s --check-prefix=NOFLAG

// FLAG-DAG: ![[RMA:.*]] = !{i32 1, !"dx.resmayalias", i32 1}
// FLAG-DAG: !llvm.module.flags = !{{{.*}}![[RMA]]{{.*}}}

// NOFLAG-NOT: dx.resmayalias
