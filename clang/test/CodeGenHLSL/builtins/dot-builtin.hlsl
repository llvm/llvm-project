// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK-LABEL: builtin_bool_to_float_type_promotion
// CHECK: %conv1 = uitofp i1 %loadedv to double
// CHECK: %hlsl.dot = fmul reassoc nnan ninf nsz arcp afn double %conv, %conv1
// CHECK: %conv2 = fptrunc reassoc nnan ninf nsz arcp afn double %hlsl.dot to float
// CHECK: ret float %conv2
float builtin_bool_to_float_type_promotion ( float p0, bool p1 ) {
  return __builtin_hlsl_dot ( (double)p0, (double)p1 );
}

// CHECK-LABEL: builtin_bool_to_float_arg1_type_promotion
// CHECK: %conv = uitofp i1 %loadedv to double
// CHECK: %conv1 = fpext reassoc nnan ninf nsz arcp afn float %1 to double
// CHECK: %hlsl.dot = fmul reassoc nnan ninf nsz arcp afn double %conv, %conv1
// CHECK: %conv2 = fptrunc reassoc nnan ninf nsz arcp afn double %hlsl.dot to float
// CHECK: ret float %conv2
float builtin_bool_to_float_arg1_type_promotion ( bool p0, float p1 ) {
  return __builtin_hlsl_dot ( (double)p0, (double)p1 );
}

// CHECK-LABEL: builtin_dot_int_to_float_promotion
// CHECK: %conv = fpext reassoc nnan ninf nsz arcp afn float %0 to double
// CHECK: %conv1 = sitofp i32 %1 to double
// CHECK: dot = fmul reassoc nnan ninf nsz arcp afn double %conv, %conv1
// CHECK: %conv2 = fptrunc reassoc nnan ninf nsz arcp afn double %hlsl.dot to float
// CHECK: ret float %conv2
float builtin_dot_int_to_float_promotion ( float p0, int p1 ) {
  return __builtin_hlsl_dot ( (double)p0, (double)p1 );
}
