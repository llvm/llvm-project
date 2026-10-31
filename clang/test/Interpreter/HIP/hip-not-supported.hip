// Check that clang-repl sets up the HIP environment but reports that HIP
// offload binary generation is not supported yet. An explicit --offload-arch
// is passed so the test does not rely on GPU auto-detection. No device code
// is run, so the architecture is arbitrary.
// FIXME: Replace this with real HIP execution tests once the HIP device parser
// can generate offload binaries.

// RUN: not clang-repl -hip --offload-arch=gfx1100 "__attribute__((device)) void f() {}" 2>&1 | FileCheck %s --implicit-check-not=error:

// CHECK: error: HIP offload binary generation is not supported yet.
