// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.0-library %s -fsyntax-only -verify=dxil
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -fsyntax-only -verify=dxil
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -fsyntax-only -verify=dxil
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -fhlsl-strict-availability \
// RUN:   -fsyntax-only -verify=dxil
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -fsyntax-only -verify=spirv
// spirv-no-diagnostics

bool shared(bool cond) {
  return QuadAll(cond); // #shared
  // dxil-error@#shared {{intrinsic 'QuadAll' is not supported in 'vertex'}}
  // dxil-error@#shared {{intrinsic 'QuadAll' is not supported in 'geometry'}}
}

bool indirect(bool cond) {
  return shared(cond);
}

[shader("vertex")]
void vertex() {
  indirect(true);
  (void)__builtin_hlsl_quad_all(true);
  // dxil-error@-1 {{intrinsic 'QuadAll' is not supported in 'vertex' shaders}}
}

[shader("vertex")]
void another_vertex() {
  indirect(false);
}

[shader("geometry")]
void geometry() {
  shared(true);
}

[shader("hull")]
void hull() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is not supported in 'hull' shaders}}
}

[shader("domain")]
void domain() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is not supported in 'domain' shaders}}
}

[shader("raygeneration")]
void raygeneration() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is not supported in 'raygeneration' shaders}}
}

[shader("intersection")]
void intersection() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is not supported in 'intersection' shaders}}
}

[shader("anyhit")]
void anyhit() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is not supported in 'anyhit' shaders}}
}

[shader("closesthit")]
void closesthit() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is not supported in 'closesthit' shaders}}
}

[shader("miss")]
void miss() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is not supported in 'miss' shaders}}
}

[shader("callable")]
void callable() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is not supported in 'callable' shaders}}
}

[shader("pixel")]
void pixel() {
  indirect(true);
}

[shader("compute")]
[numthreads(4, 2, 1)]
void compute() {
  indirect(true);
}

[shader("mesh")]
[numthreads(4, 2, 1)]
void mesh() {
  indirect(true);
}

[shader("amplification")]
[numthreads(4, 2, 1)]
void amplification() {
  indirect(true);
}

export bool exported(bool cond) {
  return shared(cond);
}

bool unused(bool cond) {
  return QuadAll(cond);
}
