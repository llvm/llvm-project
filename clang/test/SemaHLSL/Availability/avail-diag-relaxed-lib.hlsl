// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN: -fsyntax-only -Wno-error=hlsl-availability -verify %s

__attribute__((availability(shadermodel, introduced = 6.5)))
float fx(float);  // #fx

__attribute__((availability(shadermodel, introduced = 6.6)))
half fx(half);  // #fx_half

__attribute__((availability(shadermodel, introduced = 5.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.5, environment = compute)))
float fy(float); // #fy

__attribute__((availability(shadermodel, introduced = 5.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.5, environment = mesh)))
float fz(float); // #fz

float also_alive(float f) {
  // expected-warning@#also_alive_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #also_alive_fx_call
  
  // expected-warning@#also_alive_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #also_alive_fy_call

  // expected-warning@#also_alive_fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float C = fz(f); // #also_alive_fz_call

  return 0;
}

float alive(float f) {
  // expected-warning@#alive_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #alive_fx_call

  // expected-warning@#alive_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #alive_fy_call

  // expected-warning@#alive_fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float C = fz(f); // #alive_fz_call

  return also_alive(f);
}

float also_dead(float f) {
  // unreachable code - no errors expected
  float A = fx(f);
  float B = fy(f);
  float C = fz(f);
  return 0;
}

float dead(float f) {
  // unreachable code - no errors expected
  float A = fx(f);
  float B = fy(f);
  float C = fz(f);
  return also_dead(f);
}

template<typename T>
T aliveTemp(T f) {
  // expected-warning@#aliveTemp_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #aliveTemp_fx_call
  // expected-warning@#aliveTemp_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #aliveTemp_fy_call
  // expected-warning@#aliveTemp_fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float C = fz(f); // #aliveTemp_fz_call
  return 0;
}

template<typename T> T aliveTemp2(T f) {
  // expected-warning@#aliveTemp2_fx_call {{'fx' is only available on Shader Model 6.6 or newer}}
  // expected-note@#fx_half {{'fx' has been marked as being introduced in Shader Model 6.6 here, but the deployment target is Shader Model 6.0}}
  // expected-warning@#aliveTemp2_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  return fx(f); // #aliveTemp2_fx_call
}

half test(half x) {
  return aliveTemp2(x);
}

float test(float x) {
  return aliveTemp2(x);
}

class MyClass
{
  float F;
  float makeF() {
    // expected-warning@#MyClass_makeF_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
    // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
    float A = fx(F); // #MyClass_makeF_fx_call
    // expected-warning@#MyClass_makeF_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
    // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
    float B = fy(F); // #MyClass_makeF_fy_call
    // expected-warning@#MyClass_makeF_fz_call {{'fz' is unavailable}}
    // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
    float C = fz(F); // #MyClass_makeF_fz_call
    return 0;
  }
};

// Exported function without body, not used
export void exportedFunctionUnused(float f);

// Exported function with body, without export, not used
void exportedFunctionUnused(float f) {
  // expected-warning@#exportedFunctionUnused_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #exportedFunctionUnused_fx_call

  // API with shader-stage-specific availability in unused exported library function
  // - no errors expected because the actual shader stage this function
  // will be used in not known at this time
  float B = fy(f);
  float C = fz(f);
}

// Exported function with body - called from main() which is a compute shader entry point
export void exportedFunctionUsed(float f) {
  // expected-warning@#exportedFunctionUsed_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #exportedFunctionUsed_fx_call

  // expected-warning@#exportedFunctionUsed_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #exportedFunctionUsed_fy_call

  // expected-warning@#exportedFunctionUsed_fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float C = fz(f); // #exportedFunctionUsed_fz_call
}

// Shader entry point without body
[shader("compute")]
[numthreads(4,1,1)]
float main();

// Shader entry point with body
[shader("compute")]
[numthreads(4,1,1)]
float main() {
  float f = 3;
  MyClass C = { 1.0f };
  float a = alive(f);
  float b = aliveTemp<float>(f); // #aliveTemp_inst
  float c = C.makeF();
  float d = test((float)1.0);
  float e = test((half)1.0);
  exportedFunctionUsed(1.0f);
  return a * b * c;
}
