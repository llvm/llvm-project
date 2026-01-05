// RUN: %clang_dxc -all-resources-bound -T lib_6_3 -HV 202x -Vd -Xclang -emit-llvm %s | FileCheck %s --check-prefix=FLAG
// RUN: %clang_dxc -T lib_6_3 -HV 202x -Xclang -emit-llvm %s | FileCheck %s --check-prefix=NOFLAG

// FLAG-DAG: ![[ARB:.*]] = !{i32 1, !"dx.allresourcesbound", i32 1}
// FLAG-DAG: !llvm.module.flags = !{{{.*}}![[ARB]]{{.*}}}

// NOFLAG-NOT: dx.allresourcesbound
