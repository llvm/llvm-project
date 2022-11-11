// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -o - -fsyntax-only %s -verify

// expected-error@+1 {{pointers are unsupported in HLSL}}
typedef int (*fn_int)(int);
void* bark(int); // expected-error {{pointers are unsupported in HLSL}}
void meow(int*); // expected-error {{pointers are unsupported in HLSL}}

struct Foo {
  int X;
  int Y;
} *bad; // expected-error {{pointers are unsupported in HLSL}}

// expected-error@+1 {{pointers are unsupported in HLSL}}
void woof(int Foo::*Member);

int entry() {
  int X;
  Foo F;
  
  // expected-error@+2 {{the '&' operator is unsupported in HLSL}}
  // expected-error@+1 {{pointers are unsupported in HLSL}}
  int Foo::*Member = &F.X;

  
  // expected-error@+1 {{the '&' operator is unsupported in HLSL}}
  int *Y = &X; // expected-error {{pointers are unsupported in HLSL}}
  
  // expected-error@+2 {{the '->' operator is unsupported in HLSL}}
  // expected-error@+1 {{the '&' operator is unsupported in HLSL}}
  int W = (&F)->X;

  int Array[2];
  // expected-error@+1 {{the '&' operator is unsupported in HLSL}}
  *(&Array[0] + 1) = 12;
  // expected-error@+1 {{the '*' operator is unsupported in HLSL}}
  *Array = 12;
}

int roar(Foo *F) { // expected-error {{pointers are unsupported in HLSL}}
  // expected-error@+1 {{the '->' operator is unsupported in HLSL}}
  return F->X;
}

template <typename T>
void devilish_language(T look_ma_no_pointers);

void make_me_cry() {
  int X;
  // expected-error@+1 {{the '&' operator is unsupported in HLSL}}
  devilish_language(&X);

  // not-expected-error@+1 {{pointers are unsupported in HLSL}}
  devilish_language(roar); // allow function pointer decay

  // not-expected-error@+1 {{pointers are unsupported in HLSL}}
  devilish_language("roar"); // allow array pointer decay
}

struct Fish {
  struct Fins {
    int Left;
    int Right;
  };
  int X;
  int operator *() {
    return X;
  }

  // expected-note@+1 {{'->' applied to return value of the operator->() declared here}}
  Fins operator ->() {
    return Fins();
  }
};

int gone_fishing() {
  Fish F;
  int Result = *F; // user-defined dereference operators work
  // expected-error@+1 {{member reference type 'Fins' is not a pointer}}
  Result += F->Left;
  return Result;
}
