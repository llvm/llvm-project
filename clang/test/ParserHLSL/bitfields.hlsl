// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -ast-dump -x hlsl -o - %s | FileCheck %s


struct MyBitFields {
  // CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:9:3, col:25> col:16 referenced field1 'unsigned int'
  // CHECK:-ConstantExpr 0x{{[0-9a-f]+}} <col:25> 'int'
  // CHECK:-value: Int 3
  // CHECK:-IntegerLiteral 0x{{[0-9a-f]+}} <col:25> 'int' 3
  unsigned int field1 : 3; // 3 bits for field1

  // CHECK:FieldDecl 0x{{[0-9a-f]+}} <line:15:3, col:25> col:16 referenced field2 'unsigned int'
  // CHECK:-ConstantExpr 0x{{[0-9a-f]+}} <col:25> 'int'
  // CHECK:-value: Int 4
  // CHECK:-IntegerLiteral 0x{{[0-9a-f]+}} <col:25> 'int' 4
  unsigned int field2 : 4; // 4 bits for field2
  
  // CHECK:FieldDecl 0x{{[0-9a-f]+}} <line:21:3, col:16> col:7 field3 'int'
  // CHECK:-ConstantExpr 0x{{[0-9a-f]+}} <col:16> 'int'
  // CHECK:-value: Int 5
  // CHECK:-IntegerLiteral 0x{{[0-9a-f]+}} <col:16> 'int' 5
  int field3 : 5;          // 5 bits for field3 (signed)
};



[numthreads(1,1,1)]
void main() {
  MyBitFields m;
  m.field1 = 4;
  m.field2 = m.field1*2;
}