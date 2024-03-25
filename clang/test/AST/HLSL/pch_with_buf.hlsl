// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -emit-pch -o %t %S/Inputs/pch_with_buf.hlsl
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl \
// RUN:  -finclude-default-header -include-pch %t -fsyntax-only -ast-dump-all %s | FileCheck  %s

// Make sure PCH works by using function declared in PCH header.
// CHECK:FunctionDecl 0x[[FOO:[0-9a-f]+]] <{{.*}}:2:1, line:4:1> line:2:8 imported used foo 'float2 (float2, float2)'
// Make sure buffer defined in PCH works.
// CHECK:VarDecl 0x{{[0-9a-f]+}} <line:6:1, col:17> col:17 imported Buf 'RWBuffer<float>'
// Make sure declare a RWBuffer in current file works.
// CHECK:VarDecl 0x{{[0-9a-f]+}} <{{.*}}:11:1, col:23> col:23 Buf2 'hlsl::RWBuffer<float>'
hlsl::RWBuffer<float> Buf2;

float2 bar(float2 a, float2 b) {
// CHECK:CallExpr 0x{{[0-9a-f]+}} <col:10, col:18> 'float2':'float __attribute__((ext_vector_type(2)))'
// CHECK-NEXT:ImplicitCastExpr 0x{{[0-9a-f]+}} <col:10> 'float2 (*)(float2, float2)' <FunctionToPointerDecay>
// CHECK-NEXT:`-DeclRefExpr 0x{{[0-9a-f]+}} <col:10> 'float2 (float2, float2)' lvalue Function 0x[[FOO]] 'foo' 'float2 (float2, float2)'
  return foo(a, b);
}
