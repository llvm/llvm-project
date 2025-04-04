// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -finclude-default-header -verify -Wdouble-promotion -Wconversion %s

// Some helpers!
template <typename T, typename U>
struct is_same {
  static const bool value = false;
};

template <typename T>
struct is_same<T, T> {
  static const bool value = true;
};

struct SomeVals {
    int2 X;
    float2 Y;
    double2 D;
};

static SomeVals V = {1,2,3,4,5,6};

static int2 SomeArr[] = {V}; // #SomeArr
// expected-warning@#SomeArr 2 {{implicit conversion turns floating-point number into integer: 'double' to 'int'}}
// expected-warning@#SomeArr 2 {{implicit conversion turns floating-point number into integer: 'float' to 'int'}}

_Static_assert(is_same<__decltype(SomeArr), int2[3]>::value, "What is this even?");

static int2 VecArr[] = {
    int2(0,1),
    int2(2,3),
    int4(4,5,6,7),
    };

_Static_assert(is_same<__decltype(VecArr), int2[4]>::value, "One vec, two vec, three vecs, FOUR!");

static int4 V4Arr[] = {
  int2(0,1),
  int2(2,3),
};

_Static_assert(is_same<__decltype(V4Arr), int4[1]>::value, "One!");

// expected-error@+1{{too few initializers in list for type 'int4[]' (aka 'vector<int, 4>[]') (expected 4 but found 2)}}
static int4 V4ArrTooSmall[] = {
  int2(0,1),
};

// expected-error@+1{{too few initializers in list for type 'int4[]' (aka 'vector<int, 4>[]') (expected 8 but found 7)}}
static int4 V4ArrAlsoTooSmall[] = {
  int2(0,1),
  int2(2,3),
  int3(4,5,6),
};
