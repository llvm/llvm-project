// RUN: %clang_dxc -enable-16bit-types -T lib_6_3 -HV 202x -Vd -Xclang -emit-llvm %s | FileCheck %s --check-prefix=FLAG
// RUN: %clang_dxc -T lib_6_3 -HV 202x -Vd -Xclang -emit-llvm %s | FileCheck %s --check-prefix=NOFLAG

// FLAG-DAG: ![[NLP:.*]] = !{i32 1, !"dx.nativelowprec", i32 1}
// FLAG-DAG: !llvm.module.flags = !{{{.*}}![[NLP]]{{.*}}}

// NOFLAG-NOT: dx.nativelowprec
