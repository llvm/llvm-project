// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -ast-dump-filter=__is_structured_resource_element_compatible %s | FileCheck %s

// CHECK: ConceptDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> __is_structured_resource_element_compatible
// CHECK: |-TemplateTypeParmDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> referenced typename depth 0 index 0 element_type
// CHECK: `-BinaryOperator 0x{{[0-9a-f]+}} <<invalid sloc>> 'bool' lvalue '&&'
// CHECK:   |-UnaryOperator 0x{{[0-9a-f]+}} <<invalid sloc>> 'bool' lvalue prefix '!' cannot overflow
// CHECK:   | `-TypeTraitExpr 0x{{[0-9a-f]+}} <<invalid sloc>> 'bool' __builtin_hlsl_is_intangible
// CHECK:   |   `-TemplateTypeParmType 0x{{[0-9a-f]+}} 'element_type' dependent depth 0 index 0
// CHECK:   |     `-TemplateTypeParm 0x{{[0-9a-f]+}} 'element_type'
// CHECK:   `-BinaryOperator 0x{{[0-9a-f]+}} <<invalid sloc>> 'bool' lvalue '>='
// CHECK:     |-UnaryExprOrTypeTraitExpr 0x{{[0-9a-f]+}} <<invalid sloc>> 'bool' sizeof 'element_type'
// CHECK:     `-IntegerLiteral 0x{{[0-9a-f]+}} <<invalid sloc>> 'unsigned long' 1


StructuredBuffer<float> Buffer;
