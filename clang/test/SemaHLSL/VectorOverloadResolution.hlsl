// RUN: %clang_cc1 -triple dxil-unknown-shadermodel6.6-library -S -fnative-half-type -finclude-default-header -o - -ast-dump %s | FileCheck %s
void Fn(double2 D);
void Fn(half2 H);

// CHECK: FunctionDecl {{.*}} Call 'void (float2)'
// CHECK: CallExpr {{.*}}'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}}'void (double2)' lvalue Function {{.*}} 'Fn' 'void (double2)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double2':'double __attribute__((ext_vector_type(2)))' <FloatingCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2':'float __attribute__((ext_vector_type(2)))' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'float __attribute__((ext_vector_type(2)))' lvalue ParmVar {{.*}} 'F' 'float2':'float __attribute__((ext_vector_type(2)))'

void Call(float2 F) {
  Fn(F);
}

void Fn2(int64_t2 L);
void Fn2(int16_t2 S);

// CHECK: FunctionDecl {{.*}} Call2 'void (int2)'
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int64_t2)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (int64_t2)' lvalue Function {{.*}} 'Fn2' 'void (int64_t2)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t2':'long __attribute__((ext_vector_type(2)))' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int2':'int __attribute__((ext_vector_type(2)))' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int2':'int __attribute__((ext_vector_type(2)))' lvalue ParmVar {{.*}} 'I' 'int2':'int __attribute__((ext_vector_type(2)))'

void Call2(int2 I) {
  Fn2(I);
}
