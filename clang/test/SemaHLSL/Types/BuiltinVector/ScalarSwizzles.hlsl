// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library  -x hlsl \
// RUN:   -finclude-default-header -ast-dump %s | FileCheck %s


// CHECK-LABEL: ToTwoInts
// CHECK: ExtVectorElementExpr {{.*}} 'int __attribute__((ext_vector_type(2)))' xx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int __attribute__((ext_vector_type(1)))' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'V' 'int'

int2 ToTwoInts(int V){
  return V.xx;
}

// CHECK-LABEL: ToFourFloats
// CHECK: ExtVectorElementExpr {{.*}} 'float __attribute__((ext_vector_type(4)))' rrrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float __attribute__((ext_vector_type(1)))' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'V' 'float'


float4 ToFourFloats(float V){
  return V.rrrr;
}

// CHECK-LABEL: FillOne
// CHECK: ExtVectorElementExpr {{.*}} 'int __attribute__((ext_vector_type(2)))' xx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1

int2 FillOne(){
  return 1.xx;
}

// CHECK-LABEL: FillOneUnsigned
// CHECK: ExtVectorElementExpr {{.*}} 'unsigned int __attribute__((ext_vector_type(3)))' xxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1

uint3 FillOneUnsigned(){
  return 1u.xxx;
}

// CHECK-LABEL: FillOneUnsignedLong
// CHECK: ExtVectorElementExpr {{.*}} 'unsigned long __attribute__((ext_vector_type(4)))' xxxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1

vector<uint64_t,4> FillOneUnsignedLong(){
  return 1ul.xxxx;
}

// CHECK-LABEL: FillTwoPointFive
// CHECK: ExtVectorElementExpr {{.*}} 'double __attribute__((ext_vector_type(2)))' rr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 2.500000e+00

double2 FillTwoPointFive(){
  return 2.5.rr;
}

// CHECK-LABEL: FillOneHalf
// CHECK: ExtVectorElementExpr {{.*}} 'double __attribute__((ext_vector_type(3)))' rrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 5.000000e-01

double3 FillOneHalf(){
  return .5.rrr;
}

// CHECK-LABEL: FillTwoPointFiveFloat
// CHECK: ExtVectorElementExpr {{.*}} 'float __attribute__((ext_vector_type(4)))' rrrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.500000e+00

float4 FillTwoPointFiveFloat(){
  return 2.5f.rrrr;
}

// Because a signle-element accessor returns the element type rather than a
// truncated vector, this AST formulation has an initialization list to
// initialze the returned vector.

// CHECK-LABEL: FillOneHalfFloat
// CHECK: InitListExpr {{.*}} 'vector<float, 1>':'float __attribute__((ext_vector_type(1)))'
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'float' r
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 5.000000e-01

vector<float, 1> FillOneHalfFloat(){
  return .5f.r;
}

// CHECK-LABEL: HowManyFloats
// CHECK: ExtVectorElementExpr {{.*}} 'float __attribute__((ext_vector_type(2)))' rr
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'float __attribute__((ext_vector_type(2)))' rr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float __attribute__((ext_vector_type(1)))' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'V' 'float'

float2 HowManyFloats(float V) {
  return V.rr.rr;
}

// CHECK-LABEL: HooBoy
// CHECK: ExtVectorElementExpr {{.*}} 'long __attribute__((ext_vector_type(4)))' xxxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'long __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'long' 4

int64_t4 HooBoy() {
  return 4l.xxxx;
}

// This one gets a pretty wierd AST because in addition to the vector splat it
// is a double->float conversion, which results in generating an initializtion
// list with float truncation casts.

// CHECK-LABEL: AllRighty
// CHECK: InitListExpr {{.*}} 'float3':'float __attribute__((ext_vector_type(3)))'

// Vector element 0:
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'double'
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'double __attribute__((ext_vector_type(3)))' rrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0

// Vector element 1:
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'double'
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'double __attribute__((ext_vector_type(3)))' rrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1

// Vector element 2:
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'double'
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'double __attribute__((ext_vector_type(3)))' rrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double __attribute__((ext_vector_type(1)))' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2

float3 AllRighty() {
  return 1..rrr;
}
