// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: VarDecl {{.*}} specConst 'const hlsl_constant int' cinit
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 12
// CHECK-NEXT: HLSLVkConstantIdAttr {{.*}} 10    
[[vk::constant_id(10)]]
const int specConst = 12;

// CHECK: CXXRecordDecl {{.*}} implicit struct __cblayout_$Globals definition
// CHECK-NOT: FieldDecl {{.*}} specConst 'int'


