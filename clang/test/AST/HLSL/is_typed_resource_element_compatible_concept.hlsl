// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -ast-dump-filter=__is_typed_resource_element_compatible %s | FileCheck %s

// CHECK: ConceptDecl {{.*}} __is_typed_resource_element_compatible
// CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 element_type
// CHECK:  `-TypeTraitExpr {{.*}} 'bool' __builtin_hlsl_is_typed_resource_element_compatible
// CHECK:    `-TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// CHECK:      `-TemplateTypeParm {{.*}} 'element_type'

RWBuffer<float> Buffer;
