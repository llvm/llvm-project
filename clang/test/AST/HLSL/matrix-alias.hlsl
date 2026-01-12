// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -ast-dump -o - %s | FileCheck %s

// Test that matrix aliases are set up properly for HLSL

// CHECK: NamespaceDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> implicit hlsl
// CHECK-NEXT: TypeAliasTemplateDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> implicit vector
// CHECK-NEXT: TemplateTypeParmDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> class depth 0 index 0 element
// CHECK-NEXT: TemplateArgument type 'float'
// CHECK-NEXT: BuiltinType 0x{{[0-9a-fA-F]+}} 'float'
// CHECK-NEXT: NonTypeTemplateParmDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> 'int' depth 0 index 1 element_count
// CHECK-NEXT: TemplateArgument expr
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> 'int' 4
// CHECK-NEXT: TypeAliasDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> implicit vector 'vector<element, element_count>'
// CHECK-NEXT: DependentSizedExtVectorType 0x{{[0-9a-fA-F]+}} 'vector<element, element_count>' dependent <invalid sloc>
// CHECK-NEXT: TemplateTypeParmType 0x{{[0-9a-fA-F]+}} 'element' dependent depth 0 index 0
// CHECK-NEXT: TemplateTypeParm 0x{{[0-9a-fA-F]+}} 'element'
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <<invalid sloc>> 'int' lvalue
// CHECK-SAME: NonTypeTemplateParm 0x{{[0-9a-fA-F]+}} 'element_count' 'int'

// Make sure we got a using directive at the end.
// CHECK: UsingDirectiveDecl 0x{{[0-9a-fA-F]+}} <<invalid sloc>> <invalid sloc> Namespace 0x{{[0-9a-fA-F]+}} 'hlsl'

[numthreads(1,1,1)]
int entry() {
  // Verify that the alias is generated inside the hlsl namespace.
  hlsl::matrix<float, 2, 2> Mat2x2f;

  // CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:26:3, col:36>
  // CHECK-NEXT: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:29> col:29 Mat2x2f 'hlsl::matrix<float, 2, 2>'

  // Verify that you don't need to specify the namespace.
  matrix<int, 2, 2> Mat2x2i;

  // CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:32:3, col:28>
  // CHECK-NEXT: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:21> col:21 Mat2x2i 'matrix<int, 2, 2>'

  // Build a bigger matrix.
  matrix<double, 4, 4> Mat4x4d;

  // CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:38:3, col:31>
  // CHECK-NEXT: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:24> col:24 Mat4x4d 'matrix<double, 4, 4>'

  // Verify that the implicit arguments generate the correct type.
  matrix<> ImpMat4x4;

  // CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:44:3, col:21>
  // CHECK-NEXT: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:12> col:12 ImpMat4x4 'matrix<>':'matrix<float, 4, 4>'
  return 1;
}
