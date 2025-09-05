// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -finclude-default-header -verify -Wdouble-promotion -Wconversion %s

struct TwoFloats {
  float X, Y;
};

struct TwoInts {
  int Z, W;
};

struct Doggo {
  int4 LegState;
  int TailState;
  float HairCount;
  float4 EarDirection[2];
};

struct AnimalBits {
  int Legs[4];
  uint State;
  int64_t Counter;
  float4 LeftDir;
  float4 RightDir;
};

struct Kitteh {
  int4 Legs;
  int TailState;
  float HairCount;
  float4 Claws[2];
};

struct Zoo {
  Doggo Dogs[2];
  Kitteh Cats[4];
};

struct FourFloats : TwoFloats {
  float Z, W;
};

struct SlicyBits {
  int Z : 8;
  int W : 8;
};

struct ContainsResource { // #ContainsResource
  int X;
  RWBuffer<float4> B;
};

struct ContainsResourceInverted {
  RWBuffer<float4> B;
  int X;
};

void fn() {
  TwoFloats TF1 = {{{1.0, 2}}};
  TwoFloats TF2 = {1,2};
  int Val = 1;
  TwoFloats TF3 = {Val, 2}; // expected-warning{{implicit conversion from 'int' to 'float' may lose precision}}
  int2 TwoVals = 1.xx;
  int2 Something = 1.xxx; // expected-warning{{implicit conversion truncates vector: 'vector<int, 3>' (vector of 3 'int' values) to 'vector<int, 2>' (vector of 2 'int' values)}}
  TwoFloats TF4 = {TwoVals}; // expected-warning{{implicit conversion from 'int' to 'float' may lose precision}} expected-warning{{implicit conversion from 'int' to 'float' may lose precision}}

  TwoInts TI1 = {TwoVals};
  TwoInts TI2 = {TF4}; // expected-warning{{implicit conversion turns floating-point number into integer: 'float' to 'int'}} expected-warning{{implicit conversion turns floating-point number into integer: 'float' to 'int'}}

  Doggo D1 = {TI1, TI2, {Val, Val}, {{TF1, TF2}, {TF3, TF4}}}; // expected-warning{{implicit conversion from 'int' to 'float' may lose precision}}
  AnimalBits A1 = {D1}; // expected-warning{{implicit conversion turns floating-point number into integer: 'float' to 'long'}} expected-warning{{implicit conversion changes signedness: 'int' to 'unsigned int'}}

  Zoo Z1 = {D1, A1, D1, A1, D1, A1}; // #insanity

  // expected-warning@#insanity{{implicit conversion from 'int64_t' (aka 'long') to 'float' may lose precision}}
  // expected-warning@#insanity{{implicit conversion changes signedness: 'uint' (aka 'unsigned int') to 'int'}}
  // expected-warning@#insanity{{implicit conversion from 'int64_t' (aka 'long') to 'float' may lose precision}}
  // expected-warning@#insanity{{implicit conversion changes signedness: 'uint' (aka 'unsigned int') to 'int'}}
  // expected-warning@#insanity{{implicit conversion from 'int64_t' (aka 'long') to 'float' may lose precision}}
  // expected-warning@#insanity{{implicit conversion changes signedness: 'uint' (aka 'unsigned int') to 'int'}}
}

void fn2() {
  TwoFloats TF2 = {1,2};
  FourFloats FF1 = {TF2, TF2};
  FourFloats FF2 = {1,2,3,4};
  FourFloats FF3 = {1.xxx, 2};

  SlicyBits SB1 = {1,2};
  TwoInts TI1 = {SB1};
  SlicyBits SB2 = {TI1};
}

void Errs() {
  TwoFloats F1 = {}; // expected-error{{too few initializers in list for type 'TwoFloats' (expected 2 but found 0)}}
  TwoFloats F2 = {1}; // expected-error{{too few initializers in list for type 'TwoFloats' (expected 2 but found 1)}}
  TwoFloats F3 = {1,2,3}; // expected-error{{too many initializers in list for type 'TwoFloats' (expected 2 but found 3)}}

  int2 Something = {1.xxx}; // expected-error{{too many initializers in list for type 'int2' (aka 'vector<int, 2>') (expected 2 but found 3)}}
}

struct R {
  int A;
  union { // #anon
    float F;
    int4 G;
  };
};

// expected-note@#anon{{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to}}
// expected-note@#anon{{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to}}

void Err2(RWBuffer<float4> B) {
  ContainsResource RS1 = {1, B};
  ContainsResource RS2 = (1.xx); // expected-error{{no viable conversion from 'vector<int, 2>' (vector of 2 'int' values) to 'ContainsResource'}}
  ContainsResource RS3 = {B, 1}; // expected-error{{no viable conversion from 'RWBuffer<float4>' (aka 'RWBuffer<vector<float, 4>>') to 'int'}}
  ContainsResourceInverted IR = {RS1}; // expected-error{{no viable conversion from 'int' to 'hlsl::RWBuffer<vector<float, 4>>'}}

  R r = {1,2}; // expected-error{{no viable conversion from 'int' to 'R::(anonymous union at}}
}

// expected-note@#ContainsResource{{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'vector<int, 2>' (vector of 2 'int' values) to 'const ContainsResource &' for 1st argument}}
// expected-note@#ContainsResource{{candidate constructor (the implicit move constructor) not viable: no known conversion from 'vector<int, 2>' (vector of 2 'int' values) to 'ContainsResource &&' for 1st argument}}

// This note refers to the RWBuffer copy constructor that do not have a source locations
// expected-note@*{{candidate constructor not viable}}
