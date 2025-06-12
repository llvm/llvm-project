// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_dot_half
// CHECK: %hlsl.dot = fmul reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
// CHECK: ret half  %hlsl.dot
half builtin_dot_half ( half p0, half p1 ) {
  return __builtin_hlsl_dot (p0, p1 );
}

// CHECK-LABEL: builtin_dot_float
// CHECK: %hlsl.dot = fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CHECK: ret float  %hlsl.dot
float builtin_dot_float ( float p0, float p1 ) {
  return __builtin_hlsl_dot (p0, p1 );
}

// CHECK-LABEL: builtin_dot_double
// CHECK: %hlsl.dot = fmul reassoc nnan ninf nsz arcp afn double %{{.*}}, %{{.*}}
// CHECK: ret double %hlsl.dot
double builtin_dot_double( double p0, double p1 ) {
  return __builtin_hlsl_dot (p0, p1 );
}
