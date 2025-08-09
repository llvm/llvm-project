// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -ast-dump-filter=__is_structured_resource_element_compatible %s | FileCheck %s

StructuredBuffer<float> Buffer;

// CHECK: ConceptDecl {{.*}} <<invalid sloc>> <invalid sloc> __is_structured_resource_element_compatible
// CHECK-NEXT: |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> referenced typename depth 0 index 0 element_type
// CHECK-NEXT: `-BinaryOperator {{.*}} <<invalid sloc>> 'bool' lvalue '&&'
// CHECK-NEXT:   |-UnaryOperator {{.*}} <<invalid sloc>> 'bool' lvalue prefix '!' cannot overflow
// CHECK-NEXT:   | `-TypeTraitExpr {{.*}} <<invalid sloc>> 'bool' __builtin_hlsl_is_intangible
// CHECK-NEXT:   |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// CHECK-NEXT:   |     `-TemplateTypeParm {{.*}} 'element_type'
// CHECK-NEXT:   `-BinaryOperator {{.*}} <<invalid sloc>> 'bool' lvalue '>='
// CHECK-NEXT:     |-UnaryExprOrTypeTraitExpr {{.*}} <<invalid sloc>> 'bool' sizeof 'element_type'
