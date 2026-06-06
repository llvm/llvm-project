// RUN: %clang_cc1 -res-may-alias -std=hlsl202x -triple dxilv1.3-unknown-shadermodel6.3-library \
// RUN:  -finclude-default-header -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=FLAG
// RUN: %clang_cc1 -std=hlsl202x -triple dxilv1.3-unknown-shadermodel6.3-library \
// RUN:  -finclude-default-header -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=NOFLAG

// FLAG-DAG: ![[RMA:.*]] = !{i32 1, !"dx.resmayalias", i32 1}
// FLAG-DAG: !llvm.module.flags = !{{{.*}}![[RMA]]{{.*}}}

// NOFLAG-NOT: dx.resmayalias
