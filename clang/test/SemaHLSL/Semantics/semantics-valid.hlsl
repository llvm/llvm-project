// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -hlsl-entry CSMain -x hlsl  -finclude-default-header  -ast-dump -o - %s | FileCheck %s

struct s_fields {
  float a : semantic_a;
  float b : semantic_b;
// CHECK: |-CXXRecordDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-3]]:8 struct s_fields definition
// CHECK: | |-FieldDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:9 a 'float'
// CHECK: | | `-HLSLUserSemanticAttr 0x{{[0-9a-fA-F]+}} <col:13>
// CHECK: | `-FieldDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:9 b 'float'
// CHECK: |   `-HLSLUserSemanticAttr 0x{{[0-9a-fA-F]+}} <col:13>
};

float fn_foo1(float a : a, float b : b) : sem_ret { return 1.0f; }
// CHECK:      |-FunctionDecl {{.*}} <{{.*}}> col:7 fn_foo1 'float (float, float)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <{{.*}}> col:21 a 'float'
// CHECK-NEXT: | | `-HLSLUserSemanticAttr {{.*}} <{{.*}}>
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <{{.*}}> col:34 b 'float'
// CHECK-NEXT: | | `-HLSLUserSemanticAttr {{.*}} <{{.*}}>
// CHECK-NEXT: | |-CompoundStmt {{.*}} <{{.*}}>
// CHECK-NEXT: | | `-ReturnStmt {{.*}} <{{.*}}>
// CHECK-NEXT: | |   `-FloatingLiteral {{.*}} <{{.*}}> 'float' 1.000000e+00
// CHECK-NEXT: | `-HLSLUserSemanticAttr {{.*}} <{{.*}}>
float fn_foo2(float a : a, float b : b) : sem_ret : also_ret { return 1.0f; }
// CHECK:      `-FunctionDecl {{.*}} <{{.*}}> col:7 fn_foo2 'float (float, float)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <{{.*}}> col:21 a 'float'
// CHECK-NEXT:   | `-HLSLUserSemanticAttr {{.*}} <{{.*}}>
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <{{.*}}> col:34 b 'float'
// CHECK-NEXT:   | `-HLSLUserSemanticAttr {{.*}} <{{.*}}>
// CHECK-NEXT:   |-CompoundStmt {{.*}} <{{.*}}>
// CHECK-NEXT:   | `-ReturnStmt {{.*}} <{{.*}}>
// CHECK-NEXT:   |   `-FloatingLiteral {{.*}} <{{.*}}> 'float' 1.000000e+00
// CHECK-NEXT:   |-HLSLUserSemanticAttr {{.*}} <{{.*}}>
// CHECK-NEXT:   `-HLSLUserSemanticAttr {{.*}} <{{.*}}>
