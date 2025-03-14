// RUN: %clang_cc1 -res-may-alias -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=FLAG
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=NOFLAG

// FLAG-DAG: ![[RMA:.*]] = !{i32 1, !"dx.resmayalias", i32 1}
// FLAG-DAG: !llvm.module.flags = !{{{.*}}![[RMA]]{{.*}}}

// NOFLAG-NOT: dx.resmayalias
