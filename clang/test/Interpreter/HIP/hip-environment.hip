// Check that clang-repl initializes the HIP environment. HIP execution is not
// supported yet, so this only verifies that the environment is set up and that
// clang-repl reports it as unsupported. When both -cuda and -hip are passed,
// -hip wins (it appears later), so the HIP path is taken.

// RUN: not clang-repl -cuda -hip 2>&1 | FileCheck %s

// CHECK: HIP environment is initialized but not supported as of now.
