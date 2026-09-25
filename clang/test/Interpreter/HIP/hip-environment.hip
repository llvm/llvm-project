// Check that clang-repl initializes the HIP environment but reports it as
// unsupported, since HIP execution is not implemented yet. When both -cuda and
// -hip are passed, -hip wins because it appears later, so the HIP path is taken.
// An explicit --offload-arch is passed so the test does not rely on GPU
// auto-detection (--offload-arch=native), which fails on systems that have ROCm
// installed but no GPU. The test never runs device code, so the arch is
// arbitrary.

// RUN: not clang-repl -cuda -hip --offload-arch=gfx1100 2>&1 | FileCheck %s

// CHECK: HIP environment is initialized but not supported as of now.
