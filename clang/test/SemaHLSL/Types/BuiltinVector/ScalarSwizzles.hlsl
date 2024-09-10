// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library  -x hlsl \
// RUN:   -finclude-default-header -ast-dump %s | FileCheck %s


// CHECK-LABEL: ToTwoInts
// CHECK: ExtVectorElementExpr {{.*}} 'vector<int, 2>' xx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 1>' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'V' 'int'

int2 ToTwoInts(int V){
  return V.xx;
}

// CHECK-LABEL: ToFourFloats
// CHECK: ExtVectorElementExpr {{.*}} 'vector<float, 4>' rrrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'V' 'float'


float4 ToFourFloats(float V){
  return V.rrrr;
}

// CHECK-LABEL: FillOne
// CHECK: ExtVectorElementExpr {{.*}} 'vector<int, 2>' xx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1

int2 FillOne(){
  return 1.xx;
}

// CHECK-LABEL: FillOneUnsigned
// CHECK: ExtVectorElementExpr {{.*}} 'vector<unsigned int, 3>' xxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<unsigned int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1

uint3 FillOneUnsigned(){
  return 1u.xxx;
}

// CHECK-LABEL: FillOneUnsignedLong
// CHECK: ExtVectorElementExpr {{.*}} 'vector<unsigned long, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<unsigned long, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1

vector<uint64_t,4> FillOneUnsignedLong(){
  return 1ul.xxxx;
}

// CHECK-LABEL: FillTwoPointFive
// CHECK: ExtVectorElementExpr {{.*}} 'vector<double, 2>' rr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<double, 1>' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 2.500000e+00

double2 FillTwoPointFive(){
  return 2.5l.rr;
}

// CHECK-LABEL: FillOneHalf
// CHECK: ExtVectorElementExpr {{.*}} 'vector<double, 3>' rrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<double, 1>' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 5.000000e-01

double3 FillOneHalf(){
  return .5l.rrr;
}

// CHECK-LABEL: FillTwoPointFiveFloat
// CHECK: ExtVectorElementExpr {{.*}} 'vector<float, 4>' rrrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.500000e+00

float4 FillTwoPointFiveFloat(){
  return 2.5f.rrrr;
}

// Because a signle-element accessor returns the element type rather than a
// truncated vector, this AST formulation has an initialization list to
// initialze the returned vector.

// CHECK-LABEL: FillOneHalfFloat
// CHECK: ImplicitCastExpr {{.*}} 'vector<float, 1>' <VectorSplat>
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'float' r
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 5.000000e-01

vector<float, 1> FillOneHalfFloat(){
  return .5f.r;
}

// CHECK-LABEL: HowManyFloats
// CHECK: ExtVectorElementExpr {{.*}} 'vector<float, 2>' rr
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'vector<float, 2>' rr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'V' 'float'

float2 HowManyFloats(float V) {
  return V.rr.rr;
}

// CHECK-LABEL: HooBoy
// CHECK: ExtVectorElementExpr {{.*}} 'vector<long, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<long, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral {{.*}} 'long' 4

int64_t4 HooBoy() {
  return 4l.xxxx;
}

// This one gets a pretty wierd AST because in addition to the vector splat it
// is a double->float conversion, which results in generating an initializtion
// list with float truncation casts.

// CHECK-LABEL: AllRighty
// CHECK: ImplicitCastExpr {{.*}} 'vector<float, 3>' <FloatingCast>
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'vector<double, 3>' rrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<double, 1>' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00

float3 AllRighty() {
  return 1.l.rrr;
}

// CHECK-LABEL: AllRighty2
// CHECK: ExtVectorElementExpr {{.*}} 'vector<float, 3>' rrr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' <VectorSplat>
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00

float3 AllRighty2() {
  return 1..rrr;
}
