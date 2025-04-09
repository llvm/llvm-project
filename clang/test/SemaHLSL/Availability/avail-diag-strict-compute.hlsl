// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute \
// RUN: -fhlsl-strict-availability -fsyntax-only -verify %s

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
  // expected-error@#also_alive_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #also_alive_fx_call
  // expected-error@#also_alive_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #also_alive_fy_call
  // expected-error@#also_alive_fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float C = fz(f); // #also_alive_fz_call
  return 0;
}

float alive(float f) {
  // expected-error@#alive_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #alive_fx_call
  // expected-error@#alive_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #alive_fy_call
  // expected-error@#alive_fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float C = fz(f); // #alive_fz_call

  return also_alive(f);
}

float also_dead(float f) {
  // expected-error@#also_dead_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #also_dead_fx_call
  // expected-error@#also_dead_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #also_dead_fy_call
  // expected-error@#also_dead_fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float C = fz(f); // #also_dead_fz_call
  return 0;
}

float dead(float f) {
  // expected-error@#dead_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #dead_fx_call
  // expected-error@#dead_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #dead_fy_call
  // expected-error@#dead_fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float C = fz(f); // #dead_fz_call

  return also_dead(f);
}

template<typename T>
T aliveTemp(T f) {
  // expected-error@#aliveTemp_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#aliveTemp_inst {{in instantiation of function template specialization 'aliveTemp<float>' requested here}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #aliveTemp_fx_call
  // expected-error@#aliveTemp_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #aliveTemp_fy_call
  // expected-error@#aliveTemp_fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float C = fz(f); // #aliveTemp_fz_call
  return 0;
}

template<typename T> T aliveTemp2(T f) {
  // expected-error@#aliveTemp2_fx_call {{'fx' is only available on Shader Model 6.6 or newer}}
  // expected-note@#fx_half {{'fx' has been marked as being introduced in Shader Model 6.6 here, but the deployment target is Shader Model 6.0}}
  // expected-error@#aliveTemp2_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  return fx(f); // #aliveTemp2_fx_call
}

half test(half x) {
  return aliveTemp2(x); // expected-note {{in instantiation of function template specialization 'aliveTemp2<half>' requested here}}
}

float test(float x) {
  return aliveTemp2(x); // expected-note {{in instantiation of function template specialization 'aliveTemp2<float>' requested here}}
}

class MyClass
{
  float F;
  float makeF() {
    // expected-error@#MyClass_makeF_fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
    // expected-note@#fx {{'fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
    float A = fx(F); // #MyClass_makeF_fx_call
    // expected-error@#MyClass_makeF_fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
    // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
    float B = fy(F); // #MyClass_makeF_fy_call
    // expected-error@#MyClass_makeF_fz_call {{'fz' is unavailable}}
    // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 6.5 in mesh environment here, but the deployment target is Shader Model 6.0 compute environment}}
    float C = fz(F); // #MyClass_makeF_fz_call
  }
};

[numthreads(4,1,1)]
float main() {
  float f = 3;
  MyClass C = { 1.0f };
  float a = alive(f);
  float b = aliveTemp<float>(f); // #aliveTemp_inst
  float c = C.makeF();
  float d = test((float)1.0);
  float e = test((half)1.0);
  return a * b * c;
}