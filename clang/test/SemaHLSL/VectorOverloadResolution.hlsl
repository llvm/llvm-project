// RUN: %clang_cc1 -triple dxil-unknown-shadermodel6.6-library -S -fnative-half-type -finclude-default-header -o - -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECKIR
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

void Fn3( int64_t2 p0);

// CHECK: FunctionDecl {{.*}} Call3 'void (half2)'
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int64_t2)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (int64_t2)' lvalue Function {{.*}} 'Fn3' 'void (int64_t2)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t2':'long __attribute__((ext_vector_type(2)))' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half2':'half __attribute__((ext_vector_type(2)))' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'half2':'half __attribute__((ext_vector_type(2)))' lvalue ParmVar {{.*}} 'p0' 'half2':'half __attribute__((ext_vector_type(2)))'
// CHECKIR-LABEL: Call3
// CHECKIR: %conv = fptosi <2 x half> {{.*}} to <2 x i64>
void Call3(half2 p0) {
  Fn3(p0);
}

// CHECK: FunctionDecl {{.*}} Call4 'void (float2)'
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int64_t2)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (int64_t2)' lvalue Function {{.*}} 'Fn3' 'void (int64_t2)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t2':'long __attribute__((ext_vector_type(2)))' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2':'float __attribute__((ext_vector_type(2)))' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'float __attribute__((ext_vector_type(2)))' lvalue ParmVar {{.*}} 'p0' 'float2':'float __attribute__((ext_vector_type(2)))'
// CHECKIR-LABEL: Call4
// CHECKIR: {{.*}} = fptosi <2 x float> {{.*}} to <2 x i64>
void Call4(float2 p0) {
  Fn3(p0);
}

void Fn4( float2 p0);

// CHECK: FunctionDecl {{.*}} Call5 'void (int64_t2)'
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'Fn4' 'void (float2)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2':'float __attribute__((ext_vector_type(2)))' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t2':'long __attribute__((ext_vector_type(2)))' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int64_t2':'long __attribute__((ext_vector_type(2)))' lvalue ParmVar {{.*}} 'p0' 'int64_t2':'long __attribute__((ext_vector_type(2)))'
// CHECKIR-LABEL: Call5
// CHECKIR: {{.*}} = sitofp <2 x i64> {{.*}} to <2 x float>
void Call5(int64_t2 p0) {
  Fn4(p0);
}
