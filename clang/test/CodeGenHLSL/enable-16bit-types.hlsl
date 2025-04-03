// RUN: %clang_cc1 -fnative-half-type -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=FLAG
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=NOFLAG

// NOTE: -enable-16bit-types is a DXCFlag that aliases -fnative-half-type

// FLAG-DAG: ![[NLP:.*]] = !{i32 1, !"dx.nativelowprec", i32 1}
// FLAG-DAG: !llvm.module.flags = !{{{.*}}![[NLP]]{{.*}}}

// NOFLAG-NOT: dx.nativelowprec
