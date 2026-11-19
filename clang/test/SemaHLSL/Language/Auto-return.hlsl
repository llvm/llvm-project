// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202x -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -std=hlsl202y -finclude-default-header -fsyntax-only -verify %s

// expected-no-diagnostics

using namespace hlsl;

auto return_int() {
  return 42;
}

auto return_float() {
  return 1.0f;
}

extern groupshared float F;

auto return_groupshared() {
  return F;
}

auto return_const() {
  const float C = 1.0f;
  return C;
}

auto return_vector() {
  return float3(1.0f, 2.0f, 3.0f);
}

auto return_void_empty() {}

auto return_void_explicit() {
  return;
}

template <typename T> auto return_template(T Value) {
  return Value;
}

template <typename T> auto return_template_constant() {
  return 42;
}

template <typename T> struct ReturnTemplate {
  auto get() {
    return T();
  }

  T call() {
    return get();
  }
};

void use_template_member() {
  ReturnTemplate<float> Value;
  Value.call();
  _Static_assert(__is_same(__decltype(Value.get()), float),
                 "deduce class template member return type");
}

_Static_assert(__is_same(__decltype(return_int()), int), "deduce int");
_Static_assert(__is_same(__decltype(return_float()), float), "deduce float");
_Static_assert(__is_same(__decltype(return_groupshared()), float),
               "deduce float without groupshared");
_Static_assert(__is_same(__decltype(return_const()), float),
               "deduce float without const");
_Static_assert(__is_same(__decltype(return_vector()), float3), "deduce float3");
_Static_assert(__is_same(__decltype(return_void_empty()), void),
               "deduce void without return statements");
_Static_assert(__is_same(__decltype(return_void_explicit()), void),
               "deduce void from an explicit return");
_Static_assert(__is_same(__decltype(return_template(1.0f)), float),
               "deduce dependent template return type");
_Static_assert(__is_same(__decltype(return_template_constant<float>()), int),
               "deduce non-dependent template return type");
