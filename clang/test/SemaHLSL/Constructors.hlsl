// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fsyntax-only %s -verify

struct Pair {
  int First;
  int Second;

  //expected-error@+1 {{HLSL doesn't support constructors or destructors}}
  Pair() {
    First = 0;
    Second = 0;
  }

  //expected-error@+1 {{HLSL doesn't support constructors or destructors}}
  Pair(int F, int S) {
    First = F;
    Second = S;
  }

  //expected-error@+2 {{references are unsupported in HLSL}}
  //expected-error@+1 {{HLSL doesn't support constructors or destructors}}
  Pair(const Pair& P) {
    this.First = P.First;
    this.Second = P.Second;
  }

  //expected-error@+2 {{references are unsupported in HLSL}}
  //expected-error@+1 {{HLSL doesn't support constructors or destructors}}
  Pair(Pair&& P) = default;

  //expected-error@+1 {{HLSL doesn't support constructors or destructors}}
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

struct Single {
  int One;
};

void foo(Single S) {
  int A = S.One;
}

void fn() {
  //expected-error@+1 {{HLSL doesn't support constructors or functional-style casts}}
  Single S = Single();
  //expected-error@+1 {{HLSL doesn't support constructors or functional-style casts}}
  Single S2 = Single(1);
  //expected-error@+1 {{HLSL doesn't support constructors or functional-style casts}}
  Single S3 = Single(S);
  //expected-error@+1 {{HLSL doesn't support constructors or functional-style casts}}
  foo(Single(1));
  //expected-error@+1 {{HLSL doesn't support constructors or functional-style casts}}
  Single S4 = Single(1,2);
  //expected-error@+1 {{HLSL doesn't support constructors or functional-style casts}}
  Single S5 = {Single(1)};
  //expected-error@+3 {{HLSL doesn't support constructors or functional-style casts}}
  //expected-error@+2 {{HLSL doesn't support constructors or functional-style casts}}
  //expected-error@+1 {{too many initializers in list for type 'Single' (expected 1 but found 2)}}
  Single S6 = {Single(1), Single(2)};
}
