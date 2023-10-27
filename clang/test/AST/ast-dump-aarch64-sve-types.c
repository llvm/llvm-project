// Test without serialization:
// RUN: %clang_cc1 -triple aarch64-linux-gnu -ast-dump \
// RUN:   -ast-dump-filter __SV %s | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple aarch64-linux-gnu -include-pch %t \
// RUN: -ast-dump-all -ast-dump-filter __SV /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

// CHECK: TypedefDecl {{.*}} implicit __SVInt8_t '__SVInt8_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVInt8_t'

// CHECK: TypedefDecl {{.*}} implicit __SVInt16_t '__SVInt16_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVInt16_t'

// CHECK: TypedefDecl {{.*}} implicit __SVInt32_t '__SVInt32_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVInt32_t'

// CHECK: TypedefDecl {{.*}} implicit __SVInt64_t '__SVInt64_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVInt64_t'

// CHECK: TypedefDecl {{.*}} implicit __SVUint8_t '__SVUint8_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVUint8_t'

// CHECK: TypedefDecl {{.*}} implicit __SVUint16_t '__SVUint16_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVUint16_t'

// CHECK: TypedefDecl {{.*}} implicit __SVUint32_t '__SVUint32_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVUint32_t'

// CHECK: TypedefDecl {{.*}} implicit __SVUint64_t '__SVUint64_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVUint64_t'

// CHECK: TypedefDecl {{.*}} implicit __SVFloat16_t '__SVFloat16_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVFloat16_t'

// CHECK: TypedefDecl {{.*}} implicit __SVFloat32_t '__SVFloat32_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVFloat32_t'

// CHECK: TypedefDecl {{.*}} implicit __SVFloat64_t '__SVFloat64_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVFloat64_t'

// CHECK: TypedefDecl {{.*}} implicit __SVBFloat16_t '__SVBFloat16_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVBFloat16_t'

// CHECK: TypedefDecl {{.*}} implicit __SVBool_t '__SVBool_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVBool_t'

// CHECK: TypedefDecl {{.*}} implicit __SVCount_t '__SVCount_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__SVCount_t'
