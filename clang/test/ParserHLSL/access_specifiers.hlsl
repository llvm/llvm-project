// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -DSTRUCT -o - %s -verify

#ifdef STRUCT
#define KEYWORD struct
#else
#define KEYWORD class
#endif

KEYWORD Doggo {
  int legs;    // expected-note {{member is declared here}} expected-note {{member is declared here}}
protected:     // expected-warning {{access specifiers are a clang HLSL extension}}
  int ears[2]; // expected-note {{declared protected here}}
private:       // expected-warning {{access specifiers are a clang HLSL extension}}
  int tail;    // expected-note {{declared private here}} expected-note {{declared private here}}
};

KEYWORD Shiba : public Doggo { // expected-warning {{access specifiers are a clang HLSL extension}}
  int undercoat;
};

KEYWORD Akita : Doggo {
  int floof;
};

KEYWORD Chow : private Doggo {  // expected-warning {{access specifiers are a clang HLSL extension}} expected-note {{constrained by private inheritance here}}
  int megafloof;
};

KEYWORD Dachshund : protected Doggo {  // expected-warning {{access specifiers are a clang HLSL extension}} expected-note {{constrained by protected inheritance here}}
  int wiry;
};

void Puppers() {
  Shiba Shibe;
  Shibe.undercoat = 0xFFFF;
  Shibe.legs = 4;

  Shibe.tail = 1;    // expected-error {{'tail' is a private member of 'Doggo'}}
  Shibe.ears[0] = 1; // expected-error {{'ears' is a protected member of 'Doggo'}}

  Akita Aki;
  Aki.floof = 0xFFFF;
  Aki.legs = 4;

  Aki.tail = 1; // expected-error {{'tail' is a private member of 'Doggo'}}

  Chow Ch;
  Ch.megafloof = 0xFFFF;

  Ch.legs = 4; // expected-error {{'legs' is a private member of 'Doggo'}}

  Dachshund DH;
  DH.legs = 4; // expected-error {{'legs' is a protected member of 'Doggo'}}
}
