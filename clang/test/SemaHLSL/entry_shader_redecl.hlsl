// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry cs1 -o - %s -ast-dump -verify | FileCheck -DSHADERFN=cs1 -check-prefix=CHECK-ENV %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry cs2 -o - %s -ast-dump -verify | FileCheck -DSHADERFN=cs2 -check-prefix=CHECK-ENV %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry cs3 -o - %s -ast-dump -verify | FileCheck -DSHADERFN=cs3 -check-prefix=CHECK-ENV %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -ast-dump -verify | FileCheck -check-prefix=CHECK-LIB %s

// expected-no-diagnostics

// CHECK-ENV: FunctionDecl [[PROTO:0x[0-9a-f]+]] {{.*}} [[SHADERFN]] 'void ()'
// CHECK-ENV: FunctionDecl 0x{{.*}} prev [[PROTO]] {{.*}} [[SHADERFN]] 'void ()'
// CHECK-ENV-NEXT: CompoundStmt 0x
// CHECK-ENV-NEXT: HLSLNumThreadsAttr 0x
// CHECK-ENV-NEXT: HLSLShaderAttr 0x{{.*}} Implicit Compute
void cs1();
[numthreads(1,1,1)] void cs1() {}
[numthreads(1,1,1)] void cs2();
void cs2() {}
[numthreads(1,1,1)] void cs3();
[numthreads(1,1,1)] void cs3() {}

// CHECK-LIB: FunctionDecl [[PROTO:0x[0-9a-f]+]] {{.*}} s1 'void ()'
// CHECK-LIB: FunctionDecl 0x{{.*}} prev [[PROTO]] {{.*}} s1 'void ()'
// CHECK-LIB-NEXT: CompoundStmt 0x
// CHECK-LIB-NEXT: HLSLShaderAttr 0x{{.*}} Compute
// CHECK-LIB-NEXT: HLSLNumThreadsAttr 0x
void s1();
[shader("compute"), numthreads(1,1,1)] void s1() {}

// CHECK-LIB: FunctionDecl [[PROTO:0x[0-9a-f]+]] {{.*}} s2 'void ()'
// CHECK-LIB: FunctionDecl 0x{{.*}} prev [[PROTO]] {{.*}} s2 'void ()'
// CHECK-LIB-NEXT: CompoundStmt 0x
// CHECK-LIB-NEXT: HLSLShaderAttr 0x{{.*}} Compute
// CHECK-LIB-NEXT: HLSLNumThreadsAttr 0x
[shader("compute")] void s2();
[shader("compute"), numthreads(1,1,1)] void s2() {}

// CHECK-LIB: FunctionDecl [[PROTO:0x[0-9a-f]+]] {{.*}} s3 'void ()'
// CHECK-LIB: FunctionDecl 0x{{.*}} prev [[PROTO]] {{.*}} s3 'void ()'
// CHECK-LIB-NEXT: CompoundStmt 0x
// CHECK-LIB-NEXT: HLSLShaderAttr 0x{{.*}} Compute
// CHECK-LIB-NEXT: HLSLNumThreadsAttr 0x
[numthreads(1,1,1)] void s3();
[shader("compute"), numthreads(1,1,1)] void s3() {}

// CHECK-LIB: FunctionDecl [[PROTO:0x[0-9a-f]+]] {{.*}} s4 'void ()'
// CHECK-LIB: FunctionDecl 0x{{.*}} prev [[PROTO]] {{.*}} s4 'void ()'
// CHECK-LIB-NEXT: CompoundStmt 0x
// CHECK-LIB-NEXT: HLSLShaderAttr 0x{{.*}} Compute
// CHECK-LIB-NEXT: HLSLNumThreadsAttr 0x
[shader("compute"), numthreads(1,1,1)] void s4();
[shader("compute")][numthreads(1,1,1)] void s4() {}

// CHECK-LIB: FunctionDecl [[PROTO:0x[0-9a-f]+]] {{.*}} s5 'void ()'
// CHECK-LIB: FunctionDecl 0x{{.*}} prev [[PROTO]] {{.*}} s5 'void ()'
// CHECK-LIB-NEXT: CompoundStmt 0x
// CHECK-LIB-NEXT: HLSLShaderAttr 0x{{.*}} Inherited Compute
// CHECK-LIB-NEXT: HLSLNumThreadsAttr 0x
[shader("compute"), numthreads(1,1,1)] void s5();
void s5() {}

// CHECK-LIB: FunctionDecl [[PROTO:0x[0-9a-f]+]] {{.*}} s6 'void ()'
// CHECK-LIB: FunctionDecl 0x{{.*}} prev [[PROTO]] {{.*}} s6 'void ()'
// CHECK-LIB-NEXT: CompoundStmt 0x
// CHECK-LIB-NEXT: HLSLNumThreadsAttr 0x
// CHECK-LIB-NEXT: HLSLShaderAttr 0x{{.*}} Compute
[shader("compute"), numthreads(1,1,1)] void s6();
[shader("compute")] void s6() {}

// CHECK-LIB: FunctionDecl [[PROTO:0x[0-9a-f]+]] {{.*}} s7 'void ()'
// CHECK-LIB: FunctionDecl 0x{{.*}} prev [[PROTO]] {{.*}} s7 'void ()'
// CHECK-LIB-NEXT: CompoundStmt 0x
// CHECK-LIB-NEXT: HLSLShaderAttr 0x{{.*}} Inherited Compute
// CHECK-LIB-NEXT: HLSLNumThreadsAttr 0x
[shader("compute"), numthreads(1,1,1)] void s7();
[numthreads(1,1,1)] void s7() {}
