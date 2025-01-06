// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -fnative-half-type -verify %s
// expected-no-diagnostics

// Case 1: How many ways can I come up with to represent three float values?
struct ThreeFloats1 {
  float X, Y, Z;
};

struct ThreeFloats2 {
  float X[3];
};

struct ThreeFloats3 {
  float3 V;
};

struct ThreeFloats4 {
  float2 V;
  float F;
};

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(float3, float[3]), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(float3, ThreeFloats1), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(float3, ThreeFloats2), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(float3, ThreeFloats3), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(float3, ThreeFloats4), "");

// Case 2: structs and base classes and arrays, oh my!
struct Dog {
  int Leg[4];
  bool Tail;
  float Fur;
};

struct Shiba {
  int4 StubbyLegs;
  bool CurlyTail;
  struct Coating {
    float Fur;
  } F;
};

struct FourLegged {
  int FR, FL, BR, BL;
};

struct Doggo : FourLegged {
  bool WaggyBit;
  float Fuzz;
};

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(Dog, Shiba), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(Dog, Doggo), "");

// Case 3: Arrays of structs inside structs

struct Cat {
  struct Leg {
    int L;
  } Legs[4];
  struct Other {
    bool Tail;
    float Furs;
  } Bits;
};

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(Dog, Cat), "");

// case 4: Arrays of structs inside arrays of structs.
struct Pets {
  Dog Puppers[6];
  Cat Kitties[4];
};

struct Animals {
  Dog Puppers[2];
  Cat Kitties[8];
};

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(Pets, Animals), "");

// Case 5: Turtles all the way down...

typedef int Turtle;

enum Ninja : Turtle {
  Leonardo,
  Donatello,
  Michelangelo,
  Raphael,
};

enum NotNinja : Turtle {
  Fred,
  Mikey,
};

enum Mammals : uint {
  Dog,
  Cat,
};

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(Ninja, NotNinja), "");
_Static_assert(!__builtin_hlsl_is_scalarized_layout_compatible(Ninja, Mammals), "");

// Case 6: Some basic types.
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(int, int32_t), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(uint, uint32_t), "");
_Static_assert(!__builtin_hlsl_is_scalarized_layout_compatible(int, uint), "");
_Static_assert(!__builtin_hlsl_is_scalarized_layout_compatible(int, float), "");

// Even though half and float may be the same size we don't want them to be
// layout compatible since they are different types.
_Static_assert(!__builtin_hlsl_is_scalarized_layout_compatible(half, float), "");

// Case 6: Empty classes... because they're fun.

struct NotEmpty { int X; };
struct Empty {};
struct AlsoEmpty {};

struct DerivedEmpty : Empty {};

struct DerivedNotEmpty : Empty { int X; };
struct DerivedEmptyNotEmptyBase : NotEmpty {};

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(Empty, AlsoEmpty), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(Empty, DerivedEmpty), "");

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(NotEmpty, DerivedNotEmpty), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(NotEmpty, DerivedEmptyNotEmptyBase), "");
