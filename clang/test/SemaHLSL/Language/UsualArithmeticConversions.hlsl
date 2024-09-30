// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -fnative-half-type %s -DERRORS -Wconversion -Wdouble-promotion -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl2018 -finclude-default-header -fnative-half-type %s -DERRORS -Wconversion -Wdouble-promotion -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -fnative-half-type %s -ast-dump | FileCheck %s

//----------------------------------------------------------------------------//
// Case 1: float4 * int4 and inverse.
//
// In both cases here the int is converted to a float and the computation
// produces a float value.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used f4f4i4 'float4 (float4, int4)'
// CHECK: BinaryOperator {{.*}} 'float4':'vector<float, 4>' '*'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'A' 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'B' 'int4':'vector<int, 4>'
export float4 f4f4i4(float4 A, int4 B) {
  return A * B; // expected-warning{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'float4' (aka 'vector<float, 4>') may lose precision}}
}

// CHECK-LABEL: FunctionDecl {{.*}} used f4i4f4 'float4 (float4, int4)'
// CHECK: BinaryOperator {{.*}} 'float4':'vector<float, 4>' '*'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'B' 'int4':'vector<int, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'A' 'float4':'vector<float, 4>'
export float4 f4i4f4(float4 A, int4 B) {
  return B * A; // expected-warning{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'float4' (aka 'vector<float, 4>') may lose precision}}
}

//----------------------------------------------------------------------------//
// Case 2: float4 * int2 and inverse.
//
// In both cases the float vector is trunctated to a float2 and the integer
// vector is converted to a float2.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used f2f4i2 'float2 (float4, int2)'
// CHECK: BinaryOperator {{.*}} 'vector<float, 2>' '*'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 2>' <HLSLVectorTruncation>
// CHECK-NEXT: ImplicitCastExpr {{.*}}'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'A' 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 2>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int2':'vector<int, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int2':'vector<int, 2>' lvalue ParmVar {{.*}} 'B' 'int2':'vector<int, 2>'
export float2 f2f4i2(float4 A, int2 B) {
  // expected-warning@#f2f4i2 {{implicit conversion from 'int2' (aka 'vector<int, 2>') to 'vector<float, 2>' (vector of 2 'float' values) may lose precision}}
  // expected-warning@#f2f4i2 {{implicit conversion truncates vector: 'float4' (aka 'vector<float, 4>') to 'vector<float, 2>' (vector of 2 'float' values)}}
  return A * B; // #f2f4i2
}

// CHECK-LABEL: FunctionDecl {{.*}} used f2i2f4 'float2 (float4, int2)'
// CHECK: BinaryOperator {{.*}} 'vector<float, 2>' '*'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 2>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int2':'vector<int, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int2':'vector<int, 2>' lvalue ParmVar {{.*}} 'B' 'int2':'vector<int, 2>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 2>' <HLSLVectorTruncation>
// CHECK-NEXT: ImplicitCastExpr {{.*}}'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'A' 'float4':'vector<float, 4>'
export float2 f2i2f4(float4 A, int2 B) {
  // expected-warning@#f2i2f4 {{implicit conversion from 'int2' (aka 'vector<int, 2>') to 'vector<float, 2>' (vector of 2 'float' values) may lose precision}}
  // expected-warning@#f2i2f4 {{implicit conversion truncates vector: 'float4' (aka 'vector<float, 4>') to 'vector<float, 2>' (vector of 2 'float' values)}}
  return B * A; // #f2i2f4
}

//----------------------------------------------------------------------------//
// Case 3: Integers of mismatched sign, equivalent size, but the unsigned type
// has lower conversion rank.
//
// This is the odd-ball case for HLSL that isn't really in spec, but we should
// handle gracefully. The lower-ranked unsigned type is converted to the
// equivalent unsigned type of higher rank, and the signed type is also
// converted to that unsigned type (meaning `unsigned long` becomes `unsinged
// long long`, and `long long` becomes `unsigned long long`).
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used wierdo 'int4 (vector<unsigned long, 4>, vector<long long, 4>)'
// CHECK: BinaryOperator {{.*}} 'vector<unsigned long long, 4>' '*'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<unsigned long long, 4>' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<unsigned long, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr{{.*}} 'vector<unsigned long, 4>' lvalue ParmVar {{.*}} 'A' 'vector<unsigned long, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<unsigned long long, 4>' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr{{.*}}> 'vector<long long, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}}'vector<long long, 4>' lvalue ParmVar {{.*}} 'B' 'vector<long long, 4>'
export int4 wierdo(vector<unsigned long, 4> A, vector<long long, 4> B) {
  // expected-warning@#wierdo {{implicit conversion loses integer precision: 'vector<unsigned long long, 4>' (vector of 4 'unsigned long long' values) to 'vector<int, 4>' (vector of 4 'int' values)}}
  // expected-warning@#wierdo {{implicit conversion changes signedness: 'vector<long long, 4>' (vector of 4 'long long' values) to 'vector<unsigned long long, 4>' (vector of 4 'unsigned long long' values)}}
  return A * B; // #wierdo
}

//----------------------------------------------------------------------------//
// Case 4: Compound assignment of float4 with an int4.
//
// In compound assignment the RHS is converted to match the LHS.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used f4f4i4compound 'float4 (float4, int4)'
// CHECK: CompoundAssignOperator {{.*}} 'float4':'vector<float, 4>' lvalue '+=' ComputeLHSTy='float4':'vector<float, 4>' ComputeResultTy='float4':'vector<float, 4>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'A' 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'B' 'int4':'vector<int, 4>'
export float4 f4f4i4compound(float4 A, int4 B) {
  A += B; // expected-warning{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'float4' (aka 'vector<float, 4>') may lose precision}}
  return A;
}


//----------------------------------------------------------------------------//
// Case 5: Compound assignment of float2 with an int4.
//
// In compound assignment the RHS is converted to match the LHS.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used f4f2i4compound 'float4 (float2, int4)'
// CHECK: CompoundAssignOperator {{.*}} 'float2':'vector<float, 2>' lvalue '+=' ComputeLHSTy='float2':'vector<float, 2>' ComputeResultTy='float2':'vector<float, 2>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'vector<float, 2>' lvalue ParmVar {{.*}} 'A' 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2':'vector<float, 2>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 2>' <HLSLVectorTruncation>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'B' 'int4':'vector<int, 4>'
export float4 f4f2i4compound(float2 A, int4 B) {
  // expected-warning@#f4f2i4compound{{implicit conversion truncates vector: 'int4' (aka 'vector<int, 4>') to 'float2' (aka 'vector<float, 2>')}}
  // expected-warning@#f4f2i4compound{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'float2' (aka 'vector<float, 2>') may lose precision}}
  A += B; // #f4f2i4compound
  return A.xyxy;
}

//----------------------------------------------------------------------------//
// Case 6: float2 * int4
//
// The int4 vector is trunctated to int2 then converted to float2.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used f4f2i4 'float2 (float2, int4)'
// CHECK: BinaryOperator {{.*}} 'float2':'vector<float, 2>' '*'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2':'vector<float, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'vector<float, 2>' lvalue ParmVar {{.*}} 'A' 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2':'vector<float, 2>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 2>' <HLSLVectorTruncation>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'B' 'int4':'vector<int, 4>'
export float2 f4f2i4(float2 A, int4 B) {
  // expected-warning@#f4f2i4{{implicit conversion truncates vector: 'int4' (aka 'vector<int, 4>') to 'float2' (aka 'vector<float, 2>')}}
  // expected-warning@#f4f2i4{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'float2' (aka 'vector<float, 2>') may lose precision}}
  return A * B; // #f4f2i4
}

//----------------------------------------------------------------------------//
// Case 7: Compound assignment of half4 with float4, and inverse.
//
// In compound assignment the RHS is converted to match the LHS.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used f4h4f4compound 'float4 (half4, float4)'
// CHECK: CompoundAssignOperator {{.*}} 'half4':'vector<half, 4>' lvalue '+=' ComputeLHSTy='half4':'vector<half, 4>' ComputeResultTy='half4':'vector<half, 4>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'half4':'vector<half, 4>' lvalue ParmVar {{.*}} 'A' 'half4':'vector<half, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <FloatingCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'B' 'float4':'vector<float, 4>'
export float4 f4h4f4compound(half4 A, float4 B) {
  A += B; // expected-warning{{implicit conversion loses floating-point precision: 'float4' (aka 'vector<float, 4>') to 'half4' (aka 'vector<half, 4>')}}
  return B;
}

// CHECK-LABEL: FunctionDecl {{.*}} used f4f4h4compound 'float4 (float4, half4)'
// CHECK: CompoundAssignOperator {{.*}} 'float4':'vector<float, 4>' lvalue '+=' ComputeLHSTy='float4':'vector<float, 4>' ComputeResultTy='float4':'vector<float, 4>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'A' 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <FloatingCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'half4':'vector<half, 4>' lvalue ParmVar {{.*}} 'B' 'half4':'vector<half, 4>'
export float4 f4f4h4compound(float4 A, half4 B) {
  A += B; // expected-warning{{implicit conversion increases floating-point precision: 'half4' (aka 'vector<half, 4>') to 'float4' (aka 'vector<float, 4>')}}
  return A;
}

//----------------------------------------------------------------------------//
// Case 8: int64_t4 * uint4
//
// The unsigned argument is promoted to the higher ranked signed type since it
// can express all values of the unsgined argument.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used l4l4i4 'int64_t4 (int64_t4, uint4)'
// CHECK: BinaryOperator {{.*}} 'int64_t4':'vector<int64_t, 4>' '*'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t4':'vector<int64_t, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int64_t4':'vector<int64_t, 4>' lvalue ParmVar {{.*}} 'A' 'int64_t4':'vector<int64_t, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t4':'vector<int64_t, 4>' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint4':'vector<uint, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'uint4':'vector<uint, 4>' lvalue ParmVar {{.*}} 'B' 'uint4':'vector<uint, 4>'
export int64_t4 l4l4i4(int64_t4 A, uint4 B) {
  return A * B;
}

//----------------------------------------------------------------------------//
// Case 9: Compound assignment of int4 from int64_t4
//
// In compound assignment the RHS is converted to match the LHS.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used i4i4l4compound 'int4 (int4, int64_t4)'
// CHECK: CompoundAssignOperator {{.*}} 'int4':'vector<int, 4>' lvalue '+=' ComputeLHSTy='int4':'vector<int, 4>' ComputeResultTy='int4':'vector<int, 4>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'A' 'int4':'vector<int, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t4':'vector<int64_t, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int64_t4':'vector<int64_t, 4>' lvalue ParmVar {{.*}} 'B' 'int64_t4':'vector<int64_t, 4>'
export int4 i4i4l4compound(int4 A, int64_t4 B) {
  A += B; // expected-warning{{implicit conversion loses integer precision: 'int64_t4' (aka 'vector<int64_t, 4>') to 'int4' (aka 'vector<int, 4>')}}
  return A;
}

//----------------------------------------------------------------------------//
// Case 10: Compound assignment of vector<unsigned long, 4> with argument of
// vector<long long, 4>
//
// In compound assignment the RHS is converted to match the LHS. This one is
// also the weird case because it is out of spec, but we should handle it
// gracefully.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used wierdocompound 'vector<unsigned long, 4> (vector<unsigned long, 4>, vector<long long, 4>)'
// CHECK: CompoundAssignOperator {{.*}} 'vector<unsigned long, 4>' lvalue '+=' ComputeLHSTy='vector<unsigned long, 4>' ComputeResultTy='vector<unsigned long, 4>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<unsigned long, 4>' lvalue ParmVar {{.*}} 'A' 'vector<unsigned long, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<unsigned long, 4>' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<long long, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<long long, 4>' lvalue ParmVar {{.*}} 'B' 'vector<long long, 4>'
export vector<unsigned long, 4> wierdocompound(vector<unsigned long, 4> A, vector<long long, 4> B) {
  // expected-warning@#wierdocompound{{implicit conversion changes signedness: 'vector<long long, 4>' (vector of 4 'long long' values) to 'vector<unsigned long, 4>' (vector of 4 'unsigned long' values)}}
  A += B; // #wierdocompound
  return A;
}

//----------------------------------------------------------------------------//
// Case 11: Compound assignment of scalar with vector argument.
//
// Because the LHS of a compound assignment cannot change type, the RHS must be
// implicitly convertable to the LHS type.
//----------------------------------------------------------------------------//

// CHECK-LABEL: FunctionDecl {{.*}} used ffi2compound 'float (float, int2)'
// CHECK: CompoundAssignOperator {{.*}} 'float' lvalue '+=' ComputeLHSTy='float' ComputeResultTy='float'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'A' 'float'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <HLSLVectorTruncation>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int2':'vector<int, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int2':'vector<int, 2>' lvalue ParmVar {{.*}} 'B' 'int2':'vector<int, 2>'
export float ffi2compound(float A, int2 B) {
  A += B; // expected-warning {{implicit conversion turns vector to scalar: 'int2' (aka 'vector<int, 2>') to 'float'}}
  return A;
}

// CHECK-LABEL: FunctionDecl {{.*}} used iif2compound 'int (int, float2)'
// CHECK: CompoundAssignOperator {{.*}} 'int' lvalue '+=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'A' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <FloatingToIntegral>
// CHECK-NEXT: mplicitCastExpr {{.*}} 'float' <HLSLVectorTruncation>
// CHECK-NEXT: ImplicitCastExpr{{.*}} 'float2':'vector<float, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'vector<float, 2>' lvalue ParmVar {{.*}} 'B' 'float2':'vector<float, 2>'
export int iif2compound(int A, float2 B) {
  A += B; // expected-warning{{implicit conversion turns vector to scalar: 'float2' (aka 'vector<float, 2>') to 'int'}}
  return A;
}


//----------------------------------------------------------------------------//
// Case 12: Compound assignment of vector of larger size than the argument.
//
// Because the LHS of a compound assignment cannot change type, the RHS must be
// implicitly convertable to the LHS type. This fails since the RHS type can't
// be vector-extended implicitly.
//----------------------------------------------------------------------------//

#ifdef ERRORS
// The only cases that are really illegal here are when the RHS is a vector that
// is larger than the LHS or when the LHS is a scalar.

export float2 f2f4i2compound(float4 A, int2 B) {
  A += B; // expected-error{{left hand operand of type 'float4' (aka 'vector<float, 4>') to compound assignment cannot be truncated when used with right hand operand of type 'int2' (aka 'vector<int, 2>')}}
  return A.xy;
}

#endif

//----------------------------------------------------------------------------//
// Case 13: Comparison operators for mismatched arguments follow the same rules.
//
// Compare operators convert each argument following the usual arithmetic
// conversions.
//----------------------------------------------------------------------------//

// Note: these cases work and generate correct code, but the way they get there
// may change with https://github.com/llvm/llvm-project/issues/91639, because
// representing boolean vectors as 32-bit integer vectors will allow more
// efficient code generation.

// CHECK-LABEL: FunctionDecl {{.*}} used b4f4i4Compare 'bool4 (float4, int4)'
// CHECK: ImplicitCastExpr {{.*}} 'vector<bool, 4>' <IntegralToBoolean>
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<int, 4>' '<'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'A' 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'B' 'int4':'vector<int, 4>'
export bool4 b4f4i4Compare(float4 A, int4 B) {
  return A < B; // expected-warning{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'float4' (aka 'vector<float, 4>') may lose precision}}
}


// CHECK-LABEL: FunctionDecl {{.*}} used b2f2i4Compare 'bool2 (float2, int4)'
// CHECK: ImplicitCastExpr {{.*}} 'vector<bool, 2>' <IntegralToBoolean>
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<int, 2>' '<='
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2':'vector<float, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'vector<float, 2>' lvalue ParmVar {{.*}} 'A' 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2':'vector<float, 2>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 2>' <HLSLVectorTruncation>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'B' 'int4':'vector<int, 4>'

export bool2 b2f2i4Compare(float2 A, int4 B) {
  // expected-warning@#b2f2i4Compare{{implicit conversion truncates vector: 'int4' (aka 'vector<int, 4>') to 'float2' (aka 'vector<float, 2>')}}
  // expected-warning@#b2f2i4Compare{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'float2' (aka 'vector<float, 2>') may lose precision}}
  return A <= B; // #b2f2i4Compare
}

// CHECK-LABEL: FunctionDecl {{.*}} used b4fi4Compare 'bool4 (float, int4)'
// CHECK: ImplicitCastExpr {{.*}} 'vector<bool, 4>' <IntegralToBoolean>
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<int, 4>' '>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <VectorSplat>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'A' 'float'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'B' 'int4':'vector<int, 4>'
export bool4 b4fi4Compare(float A, int4 B) {
  return A > B; // expected-warning{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'vector<float, 4>' (vector of 4 'float' values) may lose precision}}
}

//----------------------------------------------------------------------------//
// Case 14: Logical operators on vectors are disallowed in HLSL 2021+
//----------------------------------------------------------------------------//

#ifdef ERRORS

#if __HLSL_VERSION >= 2021
// expected-error@#b4f4i4Logical{{invalid operands to binary expression ('float4' (aka 'vector<float, 4>') and 'int4' (aka 'vector<int, 4>'))}}
// expected-note@#b4f4i4Logical{{did you mean or?}}
#else
// expected-warning@#b4f4i4Logical{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'float4' (aka 'vector<float, 4>') may lose precision}}
#endif

export bool4 b4f4i4Logical(float4 A, int4 B) {
  return A || B; // #b4f4i4Logical
}

#if __HLSL_VERSION >= 2021
// expected-error@#b2f2i4Logical{{invalid operands to binary expression ('float2' (aka 'vector<float, 2>') and 'int4' (aka 'vector<int, 4>'))}}
// expected-note@#b2f2i4Logical{{did you mean and?}}
#else
// expected-warning@#b2f2i4Logical{{implicit conversion truncates vector: 'int4' (aka 'vector<int, 4>') to 'float2' (aka 'vector<float, 2>')}}
// expected-warning@#b2f2i4Logical{{implicit conversion from 'int4' (aka 'vector<int, 4>') to 'float2' (aka 'vector<float, 2>') may lose precision}}
#endif

export bool2 b2f2i4Logical(float2 A, int4 B) {
  return A && B; // #b2f2i4Logical
}

#if __HLSL_VERSION >= 2021
// expected-error@#b2b2b2Logical{{invalid operands to binary expression ('bool2' (aka 'vector<bool, 2>') and 'bool2')}}
// expected-note@#b2b2b2Logical{{did you mean and?}}
#endif

export bool2 b2b2b2Logical(bool2 A, bool2 B) {
  return A && B; // #b2b2b2Logical
}

#endif
