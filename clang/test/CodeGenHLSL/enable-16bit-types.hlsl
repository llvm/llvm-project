// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -std=hlsl202x -triple dxilv1.3-unknown-shadermodel6.3-library \
// RUN:  -finclude-default-header -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=FLAG
// RUN: %clang_cc1 -std=hlsl202x -triple dxilv1.3-unknown-shadermodel6.3-library \
// RUN:  -finclude-default-header -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=NOFLAG

// FLAG-DAG: ![[NLP:.*]] = !{i32 1, !"dx.nativelowprec", i32 1}
// FLAG-DAG: !llvm.module.flags = !{{{.*}}![[NLP]]{{.*}}}

// NOFLAG-NOT: dx.nativelowprec
