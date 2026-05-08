// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fsyntax-only %s -verify

struct Pair {
  int First;
  int Second;

  //expected-error@+1 {{HLSL does not support constructors or destructors}}
  Pair() {
    First = 0;
    Second = 0;
  }

  //expected-error@+1 {{HLSL does not support constructors or destructors}}
  Pair(int F, int S) {
    First = F;
    Second = S;
  }

  //expected-error@+2 {{references are unsupported in HLSL}}
  //expected-error@+1 {{HLSL does not support constructors or destructors}}
  Pair(const Pair& P) {
    this.First = P.First;
    this.Second = P.Second;
  }

  //expected-error@+2 {{references are unsupported in HLSL}}
  //expected-error@+1 {{HLSL does not support constructors or destructors}}
  Pair(Pair&& P) = default;

  //expected-error@+1 {{HLSL does not support constructors or destructors}}
  ~Pair();

  //expected-error@+2 {{references are unsupported in HLSL}}
  //expected-error@+1 {{references are unsupported in HLSL}}
  Pair& operator=(const Pair& P)
  {
    this.First = P.First;
    this.Second = P.Second;
    //expected-error@+1 {{the '*' operator is unsupported in HLSL}}
    return *this;
  }

  //expected-error@+2 {{references are unsupported in HLSL}}
  //expected-error@+1 {{references are unsupported in HLSL}}
  Pair& operator=(Pair&& P)
  {
    First = move(P.First);
    Second = move(P.Second);
    //expected-error@+1 {{the '*' operator is unsupported in HLSL}}
    return *this;
  }
};
