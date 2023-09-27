// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library  -x hlsl \
// RUN:   -finclude-default-header -ast-dump %s | FileCheck %s


// CHECK: ExtVectorElementExpr {{.*}} 'int __attribute__((ext_vector_type(2)))' xx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int __attribute__((ext_vector_type(1)))' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'V' 'int'

int2 ToTwoInts(int V){
  return V.xx;
}

// CHECK: ExtVectorElementExpr {{.*}} 'float __attribute__((ext_vector_type(4)))' rrrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float __attribute__((ext_vector_type(1)))' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'V' 'float'


float4 ToThreeFloats(float V){
  return V.rrrr;
}

// CHECK: ExtVectorElementExpr {{.*}} 'int __attribute__((ext_vector_type(2)))' xx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1

int2 FillOne(){
  return 1.xx;
}


// CHECK: ExtVectorElementExpr {{.*}} 'unsigned int __attribute__((ext_vector_type(3)))' xxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1

uint3 FillOneUnsigned(){
  return 1u.xxx;
}

// CHECK: ExtVectorElementExpr {{.*}} 'unsigned long __attribute__((ext_vector_type(4)))' xxxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1

vector<uint64_t,4> FillOneUnsignedLong(){
  return 1ul.xxxx;
}

// CHECK: ExtVectorElementExpr {{.*}} 'double __attribute__((ext_vector_type(2)))' rr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 2.500000e+00

double2 FillTwoPointFive(){
  return 2.5.rr;
}

// CHECK: ExtVectorElementExpr {{.*}} 'double __attribute__((ext_vector_type(3)))' rrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 5.000000e-01

double3 FillOneHalf(){
  return .5.rrr;
}

// CHECK: ExtVectorElementExpr {{.*}} 'float __attribute__((ext_vector_type(4)))' rrrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.500000e+00

float4 FillTwoPointFiveFloat(){
  return 2.5f.rrrr;
}

// CHECK: InitListExpr {{.*}} 'vector<float, 1>':'float __attribute__((ext_vector_type(1)))'
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'float' r
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 5.000000e-01

vector<float, 1> FillOneHalfFloat(){
  return .5f.r;
}
