// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s
// CHECK: NamespaceDecl {{.*}} implicit hlsl
// CHECK: TypeAliasTemplateDecl {{.*}} implicit vector
// CHECK-NEXT: TemplateTypeParmDecl {{.*}} class depth 0 index 0 element
// CHECK-NEXT: TemplateArgument type 'float'
// CHECK-NEXT: BuiltinType {{.*}} 'float'
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} 'int' depth 0 index 1 element_count
// CHECK-NEXT: TemplateArgument expr
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: TypeAliasDecl {{.*}} implicit vector 'vector<element, element_count>'
// CHECK-NEXT: DependentSizedExtVectorType {{.*}} 'vector<element, element_count>' dependent
// CHECK-NEXT: TemplateTypeParmType {{.*}} 'element' dependent depth 0 index 0
// CHECK-NEXT: TemplateTypeParm {{.*}} 'element'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue
// NonTypeTemplateParm {{.*}} 'element_count' 'int'

// Make sure we got a using directive at the end.
// CHECK: UsingDirectiveDecl {{.*}} Namespace {{.*}} 'hlsl'

[numthreads(1,1,1)]
int entry() {
  // Verify that the alias is generated inside the hlsl namespace.
  hlsl::vector<float, 2> Vec2 = {1.0, 2.0};

  // CHECK: DeclStmt
  // CHECK-NEXT: VarDecl {{.*}} Vec2 'hlsl::vector<float, 2>':'vector<float, 2>' cinit

  // Verify that you don't need to specify the namespace.
  vector<int, 2> Vec2a = {1, 2};

  // CHECK: DeclStmt
  // CHECK-NEXT: VarDecl {{.*}} Vec2a 'vector<int, 2>' cinit

  // Build a bigger vector.
  vector<double, 4> Vec4 = {1.0, 2.0, 3.0, 4.0};

  // CHECK: DeclStmt
  // CHECK-NEXT: VarDecl {{.*}} used Vec4 'vector<double, 4>' cinit

  // Verify that swizzles still work.
  vector<double, 3> Vec3 = Vec4.xyz;

  // CHECK: DeclStmt {{.*}}
  // CHECK-NEXT: VarDecl {{.*}} Vec3 'vector<double, 3>' cinit

  // Verify that the implicit arguments generate the correct type.
  vector<> ImpVec4 = {1.0, 2.0, 3.0, 4.0};

  // CHECK: DeclStmt
  // CHECK-NEXT: VarDecl {{.*}} ImpVec4 'vector<>':'vector<float, 4>' cinit
  return 1;
}
