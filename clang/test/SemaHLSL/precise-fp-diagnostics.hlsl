// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fsyntax-only -Werror -DPRECISE_AFTER -verify=precise %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fsyntax-only -Werror -verify=fast %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fsyntax-only -Werror -fno-fast-math -verify=precise %s
// precise-no-diagnostics

bool check_nan(float x) {
  return __builtin_hlsl_elementwise_isnan(x); // fast-error {{use of NaN is undefined behavior}}
}

bool check_inf(float x) {
  return __builtin_hlsl_elementwise_isinf(x); // fast-error {{use of infinity is undefined behavior}}
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
bool ignored_nan(float x) {
  return __builtin_hlsl_elementwise_isnan(x);
}
#pragma clang diagnostic pop

#ifdef PRECISE_AFTER
typedef precise float precise_after;
#endif
