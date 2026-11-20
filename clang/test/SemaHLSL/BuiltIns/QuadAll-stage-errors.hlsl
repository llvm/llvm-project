// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.0-library %s -fsyntax-only -verify=dxil,common
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -fsyntax-only -verify=dxil,common
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -fsyntax-only -verify=dxil,common
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -fhlsl-strict-availability \
// RUN:   -fsyntax-only -verify=dxil,common
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -Wno-error=hlsl-availability \
// RUN:   -fsyntax-only -verify=relaxed,common
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -fsyntax-only -verify=spirv
// spirv-no-diagnostics
// common-note@* 10 {{'QuadAll' has been marked as being introduced}}

bool shared(bool cond) {
  return QuadAll(cond); // #shared
  // dxil-error@#shared 2 {{'QuadAll' is unavailable}}
  // relaxed-warning@#shared 2 {{'QuadAll' is unavailable}}
}

bool indirect(bool cond) {
  return shared(cond);
}

[shader("vertex")]
void vertex() {
  indirect(true);
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
  // dxil-error@-1 {{'QuadAll' is unavailable}}
  // relaxed-warning@-2 {{'QuadAll' is unavailable}}
}

[shader("domain")]
void domain() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is unavailable}}
  // relaxed-warning@-2 {{'QuadAll' is unavailable}}
}

[shader("raygeneration")]
void raygeneration() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is unavailable}}
  // relaxed-warning@-2 {{'QuadAll' is unavailable}}
}

[shader("intersection")]
void intersection() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is unavailable}}
  // relaxed-warning@-2 {{'QuadAll' is unavailable}}
}

[shader("anyhit")]
void anyhit() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is unavailable}}
  // relaxed-warning@-2 {{'QuadAll' is unavailable}}
}

[shader("closesthit")]
void closesthit() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is unavailable}}
  // relaxed-warning@-2 {{'QuadAll' is unavailable}}
}

[shader("miss")]
void miss() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is unavailable}}
  // relaxed-warning@-2 {{'QuadAll' is unavailable}}
}

[shader("callable")]
void callable() {
  (void)QuadAll(true);
  // dxil-error@-1 {{'QuadAll' is unavailable}}
  // relaxed-warning@-2 {{'QuadAll' is unavailable}}
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
