// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-library -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK-LABLE:FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> w0 'void ()'
// CHECK:HLSLWaveSizeAttr 0x{{[0-9a-f]+}} <{{.*}}> 128 0 0
 [numthreads(8,8,1)]
 [WaveSize(128)]
 void w0() {
 }



// CHECK-LABLE:FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> w1 'void ()'
// CHECK:HLSLWaveSizeAttr 0x{{[0-9a-f]+}} <{{.*}}> 8 64 0
 [numthreads(8,8,1)]
 [WaveSize(8, 64)]
 void w1() {
 }


// CHECK-LABLE:FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> w2 'void ()'
// CHECK:HLSLWaveSizeAttr 0x{{[0-9a-f]+}} <{{.*}}> 8 128 64
// Duplicate WaveSize attribute will be ignored.
// CHECK-NOT:HLSLWaveSizeAttr 0x{{[0-9a-f]+}} <{{.*}}> 8 128 64
 [numthreads(8,8,1)]
 [WaveSize(8, 128, 64)]
 [WaveSize(8, 128, 64)]
 void w2() {
 }
