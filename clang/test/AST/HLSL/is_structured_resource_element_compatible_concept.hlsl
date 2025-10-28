// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -ast-dump-filter=__is_structured_resource_element_compatible %s | FileCheck %s

// CHECK: ConceptDecl {{.*}} __is_structured_resource_element_compatible
// CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 element_type
// CHECK: `-BinaryOperator {{.*}} 'bool' lvalue '&&'
// CHECK:   |-UnaryOperator {{.*}} 'bool' lvalue prefix '!' cannot overflow
// CHECK:   | `-TypeTraitExpr {{.*}} 'bool' __builtin_hlsl_is_intangible
// CHECK:   |   `-TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// CHECK:   |     `-TemplateTypeParm {{.*}} 'element_type'
// CHECK:   `-BinaryOperator {{.*}} 'bool' lvalue '>='
// CHECK:     |-UnaryExprOrTypeTraitExpr {{.*}} 'bool' sizeof 'element_type'
// CHECK:     `-IntegerLiteral {{.*}} '__size_t':'unsigned long' 1


StructuredBuffer<float> Buffer;
