// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.0-library %s -fsyntax-only -verify
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -fsyntax-only -verify \
// RUN:   -fhlsl-strict-availability
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -fsyntax-only -verify
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -fsyntax-only -verify

bool test_quad_all(bool cond) {
  return QuadAll(cond); // #quad
}

export bool exported_quad_all(bool cond) {
  return test_quad_all(cond);
}

[shader("pixel")]
void pixel_main() {
  test_quad_all(true);
  (void)__builtin_hlsl_quad_all(true);
}

[shader("compute")]
[numthreads(2, 2, 1)]
void compute_main() {
  test_quad_all(true);
  (void)__builtin_hlsl_quad_all(true);
}

[shader("mesh")]
[numthreads(2, 2, 1)]
void mesh_main() {
  test_quad_all(true);
}

[shader("amplification")]
[numthreads(2, 2, 1)]
void amplification_main() {
  test_quad_all(true);
}

// expected-error@#quad {{'QuadAll' is not supported in 'vertex' shaders, \
requires one of the following: pixel, compute, mesh, amplification}}
[shader("vertex")]
void vertex_main() {
  test_quad_all(true);
  // expected-error@+3 {{'__builtin_hlsl_quad_all' is not supported in \
'vertex' shaders, requires one of the following: \
pixel, compute, mesh, amplification}}
  (void)__builtin_hlsl_quad_all(true);
}

// expected-error@#quad {{'QuadAll' is not supported in 'geometry' shaders, \
requires one of the following: pixel, compute, mesh, amplification}}
[shader("geometry")]
void geometry_main() { test_quad_all(true); }

// expected-error@#quad {{'QuadAll' is not supported in 'hull' shaders, \
requires one of the following: pixel, compute, mesh, amplification}}
[shader("hull")]
void hull_main() { test_quad_all(true); }

// expected-error@#quad {{'QuadAll' is not supported in 'domain' shaders, \
requires one of the following: pixel, compute, mesh, amplification}}
[shader("domain")]
void domain_main() { test_quad_all(true); }

// expected-error@#quad {{'QuadAll' is not supported in 'raygeneration' \
shaders, requires one of the following: pixel, compute, mesh, amplification}}
[shader("raygeneration")]
void raygeneration_main() { test_quad_all(true); }

// expected-error@#quad {{'QuadAll' is not supported in 'intersection' \
shaders, requires one of the following: pixel, compute, mesh, amplification}}
[shader("intersection")]
void intersection_main() { test_quad_all(true); }

// expected-error@#quad {{'QuadAll' is not supported in 'anyhit' shaders, \
requires one of the following: pixel, compute, mesh, amplification}}
[shader("anyhit")]
void anyhit_main() { test_quad_all(true); }

// expected-error@#quad {{'QuadAll' is not supported in 'closesthit' shaders, \
requires one of the following: pixel, compute, mesh, amplification}}
[shader("closesthit")]
void closesthit_main() { test_quad_all(true); }

// expected-error@#quad {{'QuadAll' is not supported in 'miss' shaders, \
requires one of the following: pixel, compute, mesh, amplification}}
[shader("miss")]
void miss_main() { test_quad_all(true); }

// expected-error@#quad {{'QuadAll' is not supported in 'callable' shaders, \
requires one of the following: pixel, compute, mesh, amplification}}
[shader("callable")]
void callable_main() { test_quad_all(true); }
