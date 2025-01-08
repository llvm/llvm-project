// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.5-library -fsyntax-only -verify %s


void f1(void) __attribute__((availability(shadermodel, introduced = 6.0, environment="pixel"))); // expected-error {{expected an environment name, e.g., 'compute'}}

void f2(void) __attribute__((availability(shadermodel, introduced = 6.0, environment=pixel, environment=compute))); // expected-error {{redundant 'environment' availability change; only the last specified change will be used}}

void f3(void) __attribute__((availability(shadermodel, strict, introduced = 6.0, environment = mesh))); // expected-error {{unexpected parameter 'strict' in availability attribute, not permitted in HLSL}}

int main() {
}
