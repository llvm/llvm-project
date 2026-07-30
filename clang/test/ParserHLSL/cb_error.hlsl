// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

// expected-error@+2 {{expected identifier}}
// expected-error@+1 {{expected unqualified-id}}
cbuffer { ... };
// expected-error@+1 {{expected '{'}}
cbuffer missing_definition;
// expected-error@+1 {{expected unqualified-id}}
int cbuffer;
// expected-error@+1 {{expected identifier}}
cbuffer;

// expected-error@+2 {{expected identifier}}
// expected-error@+1 {{expected unqualified-id}}
tbuffer { ... };
// expected-error@+1 {{expected '{'}}
tbuffer missing_definition;
// expected-error@+1 {{expected unqualified-id}}
int tbuffer;
// expected-error@+1 {{expected identifier}}
tbuffer;

// expected-error@+1 {{expected unqualified-id}}
cbuffer A {}, B{}

// cbuffer inside namespace is supported.
namespace N {
  cbuffer A {
    float g;
  }
}

cbuffer A {
  // expected-error@+1 {{invalid declaration inside cbuffer}}
  namespace N {
  }
}

cbuffer A {
  // expected-error@+1 {{invalid declaration inside cbuffer}}
  cbuffer Nested {
  }
}

struct S {
  // expected-error@+1 {{expected member name or ';' after declaration specifiers}}
  cbuffer what {
    int y;
  }
};

void func() {
  // expected-error@+1 {{expected expression}}
  tbuffer derp {
    int z;
  }

  decltype(derp) another {
    int a;
  }
}

// struct decl inside cb is supported.
cbuffer A {
  struct S2 {
    float s;
  };
  S2 s;
}

// function decl inside cb is supported.
cbuffer A {
  float foo_inside_cb() { return 1.2;}
}
