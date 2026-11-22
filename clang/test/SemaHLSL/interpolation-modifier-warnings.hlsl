// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -Wno-duplicate-decl-specifier -verify=override %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -Wno-ignored-attributes -verify=duplicate %s

// The existing diagnostic groups independently control duplicates and overrides.
// expected-warning@+2 {{duplicate interpolation modifier 'linear'}}
// duplicate-warning@+1 {{duplicate interpolation modifier 'linear'}}
void duplicates(linear linear float x);

// expected-warning@+2 {{interpolation modifier 'sample' overrides 'center'}}
// override-warning@+1 {{interpolation modifier 'sample' overrides 'center'}}
void locations(center sample float x);
