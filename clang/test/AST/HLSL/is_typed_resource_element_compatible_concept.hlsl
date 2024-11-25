// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -ast-dump-filter=__is_typed_resource_element_compatible %s | FileCheck %s

// CHECK: ConceptDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> __is_typed_resource_element_compatible
// CHECK: |-TemplateTypeParmDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> referenced typename depth 0 index 0 element_type
// CHECK:  `-TypeTraitExpr 0x{{[0-9a-f]+}} <<invalid sloc>> 'bool' __builtin_hlsl_is_typed_resource_element_compatible
// CHECK:    `-TemplateTypeParmType 0x{{[0-9a-f]+}} 'element_type' dependent depth 0 index 0
// CHECK:      `-TemplateTypeParm 0x{{[0-9a-f]+}} 'element_type'

RWBuffer<float> Buffer;
