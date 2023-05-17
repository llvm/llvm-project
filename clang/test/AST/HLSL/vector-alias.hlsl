// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s 

// CHECK: NamespaceDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> implicit hlsl
// CHECK-NEXT: TypeAliasTemplateDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> implicit vector
// CHECK-NEXT: TemplateTypeParmDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> class depth 0 index 0 element
// CHECK-NEXT: TemplateArgument type 'float'
// CHECK-NEXT: BuiltinType 0x{{[0-9a-fA-F]+}} 'float'
// CHECK-NEXT: NonTypeTemplateParmDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> 'int' depth 0 index 1 element_count
// CHECK-NEXT: TemplateArgument expr
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> 'int' 4
// CHECK-NEXT: TypeAliasDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> implicit vector 'element __attribute__((ext_vector_type(element_count)))'
// CHECK-NEXT: DependentSizedExtVectorType 0x{{[0-9a-fA-F]+}} 'element __attribute__((ext_vector_type(element_count)))' dependent <invalid sloc>
// CHECK-NEXT: TemplateTypeParmType 0x{{[0-9a-fA-F]+}} 'element' dependent depth 0 index 0
// CHECK-NEXT: TemplateTypeParm 0x{{[0-9a-fA-F]+}} 'element'
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <<invalid sloc>> 'int' lvalue
// NonTypeTemplateParm 0x{{[0-9a-fA-F]+}} 'element_count' 'int'

// Make sure we got a using directive at the end.
// CHECK: UsingDirectiveDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> Namespace 0x{{[0-9a-fA-F]+}} 'hlsl'

[numthreads(1,1,1)]
int entry() {
  // Verify that the alias is generated inside the hlsl namespace.
  hlsl::vector<float, 2> Vec2 = {1.0, 2.0};

  // CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:24:3, col:43>
  // CHECK-NEXT: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:42> col:26 Vec2 'hlsl::vector<float, 2>':'float __attribute__((ext_vector_type(2)))' cinit

  // Verify that you don't need to specify the namespace.
  vector<int, 2> Vec2a = {1, 2};

  // CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:30:3, col:32>
  // CHECK-NEXT: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:31> col:18 Vec2a 'vector<int, 2>':'int __attribute__((ext_vector_type(2)))' cinit

  // Build a bigger vector.
  vector<double, 4> Vec4 = {1.0, 2.0, 3.0, 4.0};

  // CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:36:3, col:48>
  // CHECK-NEXT: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:47> col:21 used Vec4 'vector<double, 4>':'double __attribute__((ext_vector_type(4)))' cinit

  // Verify that swizzles still work.
  vector<double, 3> Vec3 = Vec4.xyz;

  // CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:42:3, col:36>
  // CHECK-NEXT: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:33> col:21 Vec3 'vector<double, 3>':'double __attribute__((ext_vector_type(3)))' cinit

  // Verify that the implicit arguments generate the correct type.
  vector<> ImpVec4 = {1.0, 2.0, 3.0, 4.0};

  // CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:48:3, col:42>
  // CHECK-NEXT: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:41> col:12 ImpVec4 'vector<>':'float __attribute__((ext_vector_type(4)))' cinit
  return 1;
}
