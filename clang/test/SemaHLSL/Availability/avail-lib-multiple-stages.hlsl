// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN: -fsyntax-only -verify %s

__attribute__((availability(shadermodel, introduced = 6.5)))
float fx(float);  // #fx

__attribute__((availability(shadermodel, introduced = 5.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.5, environment = compute)))
float fy(float); // #fy

__attribute__((availability(shadermodel, introduced = 5.0, environment = compute)))
float fz(float); // #fz


void F(float f) {
  // Make sure we only get this error once, even though this function is scanned twice - once
  // in compute shader context and once in pixel shader context.
  // expected-error@#fx_call {{'fx' is only available on Shader Model 6.5 or newer}}
  // expected-note@#fx {{fx' has been marked as being introduced in Shader Model 6.5 here, but the deployment target is Shader Model 6.0}}
  float A = fx(f); // #fx_call
  
  // expected-error@#fy_call {{'fy' is only available in compute environment on Shader Model 6.5 or newer}}
  // expected-note@#fy {{'fy' has been marked as being introduced in Shader Model 6.5 in compute environment here, but the deployment target is Shader Model 6.0 compute environment}}
  float B = fy(f); // #fy_call

  // expected-error@#fz_call {{'fz' is unavailable}}
  // expected-note@#fz {{'fz' has been marked as being introduced in Shader Model 5.0 in compute environment here, but the deployment target is Shader Model 6.0 pixel environment}}
  float X = fz(f); // #fz_call
}

void deadCode(float f) {
  // no diagnostics expected under default diagnostic mode
  float A = fx(f);
  float B = fy(f);
  float X = fz(f);
}

// Pixel shader
[shader("pixel")]
void mainPixel() {
  F(1.0);
}

// First Compute shader
[shader("compute")]
[numthreads(4,1,1)]
void mainCompute1() {
  F(2.0);
}

// Second compute shader to make sure we do not get duplicate messages if F is called
// from multiple entry points.
[shader("compute")]
[numthreads(4,1,1)]
void mainCompute2() {
  F(3.0);
}
