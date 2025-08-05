// RUN: %clang_cc1 -triple spirv-unknown-vulkan-library -x hlsl \
// RUN:  -finclude-default-header -emit-pch -o %t %S/Inputs/pch_spirv_type.hlsl
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-library -x hlsl \
// RUN:  -finclude-default-header -include-pch %t -ast-dump-all %s \
// RUN: | FileCheck  %s

// Make sure PCH works by using function declared in PCH header and declare a SpirvType in current file.
// CHECK:FunctionDecl 0x[[FOO:[0-9a-f]+]] <{{.*}}:2:1, line:4:1> line:2:8 imported used foo 'float2 (float2, float2)'
// CHECK:VarDecl 0x{{[0-9a-f]+}} <{{.*}}:10:1, col:92> col:92 buffers2 'hlsl_constant vk::SpirvOpaqueType<28, RWBuffer<float>, vk::integral_constant<uint, 4>>':'hlsl_constant __hlsl_spirv_type<28, 0, 0, RWBuffer<float>, vk::integral_constant<unsigned int, 4>>'
vk::SpirvOpaqueType</* OpTypeArray */ 28, RWBuffer<float>, vk::integral_constant<uint, 4>> buffers2;

float2 bar(float2 a, float2 b) {
// CHECK:CallExpr 0x{{[0-9a-f]+}} <col:10, col:18> 'float2':'vector<float, 2>'
// CHECK-NEXT:ImplicitCastExpr 0x{{[0-9a-f]+}} <col:10> 'float2 (*)(float2, float2)' <FunctionToPointerDecay>
// CHECK-NEXT:`-DeclRefExpr 0x{{[0-9a-f]+}} <col:10> 'float2 (float2, float2)' lvalue Function 0x[[FOO]] 'foo' 'float2 (float2, float2)'
  return foo(a, b);
}
