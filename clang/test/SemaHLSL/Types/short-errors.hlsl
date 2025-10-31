// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-compute -verify %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.8-compute -verify %s

void asArg(inout short F) { F + 1;}
// expected-error@-1 {{unknown type name short}}

export void asVarDecl() {
  short A = 1;
  // expected-error@-1 {{unknown type name short}}  
  fn(A);
}

export short asReturnType() {
// expected-error@-1 {{unknown type name short}}
  return 1;
}

struct S {
  short A;
  // expected-error@-1 {{unknown type name short}}
};
