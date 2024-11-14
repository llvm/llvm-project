// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -fsyntax-only %s -DERROR=1 -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -ast-dump %s | FileCheck %s


// Case 1: Prioritize splat without conversion over conversion. In this case the
// called functions have valid overloads for each type, however one of the
// overloads is a vector rather than scalar. Each call should resolve to the
// same type, and the vector should splat.
void HalfFloatDoubleV(double2 D);
void HalfFloatDoubleV(float F);
void HalfFloatDoubleV(half H);

void HalfFloatVDouble(double D);
void HalfFloatVDouble(float2 F);
void HalfFloatVDouble(half H);

void HalfVFloatDouble(double D);
void HalfVFloatDouble(float F);
void HalfVFloatDouble(half2 H);


// CHECK-LABEL: FunctionDecl {{.*}} Case1 'void (half, float, double)'
void Case1(half H, float F, double D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half)' lvalue Function {{.*}} 'HalfFloatDoubleV' 'void (half)'
  HalfFloatDoubleV(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float)' lvalue Function {{.*}} 'HalfFloatDoubleV' 'void (float)'
  HalfFloatDoubleV(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'HalfFloatDoubleV' 'void (double2)'
  HalfFloatDoubleV(D);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half)' lvalue Function {{.*}} 'HalfFloatVDouble' 'void (half)'
  HalfFloatVDouble(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'HalfFloatVDouble' 'void (float2)'
  HalfFloatVDouble(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'HalfFloatVDouble' 'void (double)'
  HalfFloatVDouble(D);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half2)' lvalue Function {{.*}} 'HalfVFloatDouble' 'void (half2)'
  HalfVFloatDouble(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float)' lvalue Function {{.*}} 'HalfVFloatDouble' 'void (float)'
  HalfVFloatDouble(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'HalfVFloatDouble' 'void (double)'
  HalfVFloatDouble(D);
}

// Case 2: Prefer splat+promotion over conversion. In this case the overloads
// require a splat+promotion or a conversion. The call will resolve to the
// splat+promotion.
void HalfDoubleV(double2 D);
void HalfDoubleV(half H);

// CHECK-LABEL: FunctionDecl {{.*}} Case2 'void (float)'
void Case2(float F) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'HalfDoubleV' 'void (double2)'
  HalfDoubleV(F);
}

// Case 3: Prefer promotion or conversion without splat over the splat. In this
// case the scalar value will overload to the scalar function.
void DoubleV(double D);
void DoubleV(double2 V);

void HalfV(half D);
void HalfV(half2 V);

// CHECK-LABEL: FunctionDecl {{.*}} Case3 'void (float)'
void Case3(float F) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'DoubleV' 'void (double)'
  DoubleV(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half)' lvalue Function {{.*}} 'HalfV' 'void (half)'
  HalfV(F);
}

#if ERROR
// Case 4: It is ambiguous to resolve two splat+conversion or splat+promotion
// functions. In all the calls below an error occurs.
void FloatVDoubleV(float2 F); // expected-note {{candidate function}}
void FloatVDoubleV(double2 D); // expected-note {{candidate function}}

void HalfVFloatV(half2 H); // expected-note {{candidate function}}
void HalfVFloatV(float2 F); // expected-note {{candidate function}}

void Case4(half H, double D) {
  FloatVDoubleV(H); // expected-error {{call to 'FloatVDoubleV' is ambiguous}}

  HalfVFloatV(D); // expected-error {{call to 'HalfVFloatV' is ambiguous}}
}

// Case 5: It is ambiguous to resolve two splats of different lengths.
void FloatV(float2 V); // expected-note {{candidate function}} expected-note {{candidate function}} expected-note {{candidate function}}
void FloatV(float4 V); // expected-note {{candidate function}} expected-note {{candidate function}} expected-note {{candidate function}}

void Case5(half H, float F, double D) {
  FloatV(H); // expected-error {{call to 'FloatV' is ambiguous}}
  FloatV(F); // expected-error {{call to 'FloatV' is ambiguous}}
  FloatV(D); // expected-error {{call to 'FloatV' is ambiguous}}
}
#endif

// Case 5: Vectors truncate or match, but don't extend.
void FloatV24(float2 V);
void FloatV24(float4 V);

// CHECK-LABEL: FunctionDecl {{.*}} Case5 'void (half3, float3, double3, half4, float4, double4)'
void Case5(half3 H3, float3 F3, double3 D3, half4 H4, float4 F4, double4 D4) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'FloatV24' 'void (float2)'
  FloatV24(H3); // expected-warning{{implicit conversion truncates vector: 'half3' (aka 'vector<half, 3>') to 'vector<float, 2>' (vector of 2 'float' values)}}

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'FloatV24' 'void (float2)'
  FloatV24(F3); // expected-warning{{implicit conversion truncates vector: 'float3' (aka 'vector<float, 3>') to 'vector<float, 2>' (vector of 2 'float' values)}}

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'FloatV24' 'void (float2)'
  FloatV24(D3); // expected-warning{{implicit conversion truncates vector: 'double3' (aka 'vector<double, 3>') to 'vector<float, 2>' (vector of 2 'float' values)}}

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float4)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float4)' lvalue Function {{.*}} 'FloatV24' 'void (float4)'
  FloatV24(H4);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float4)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float4)' lvalue Function {{.*}} 'FloatV24' 'void (float4)'
  FloatV24(F4);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float4)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float4)' lvalue Function {{.*}} 'FloatV24' 'void (float4)'
  FloatV24(D4);
}
