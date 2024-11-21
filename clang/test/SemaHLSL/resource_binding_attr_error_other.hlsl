// RUN: not %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s  | FileCheck %s

// XFAIL: *
// This expectedly fails because RayQuery is an unsupported type.
// When it becomes supported, we should expect an error due to 
// the variable type being classified as "other", and according
// to the spec, err_hlsl_unsupported_register_type_and_variable_type
// should be emitted.
RayQuery<0> r1: register(t0);
