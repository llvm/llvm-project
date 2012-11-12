// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test template instantiation for C99-specific features.

// ---------------------------------------------------------------------
// Designated initializers
// ---------------------------------------------------------------------
template<typename T, typename XType, typename YType>
struct DesigInit0 {
  void f(XType x, YType y) {
    T agg = { 
      .y = y, // expected-error{{does not refer}}
      .x = x  // expected-error{{does not refer}}
    };
  }
};

struct Point2D {
  float x, y;
};

template struct DesigInit0<Point2D, int, double>;

struct Point3D {
  float x, y, z;
};

template struct DesigInit0<Point3D, int, double>;

struct Color { 
  unsigned char red, green, blue;
};

struct ColorPoint3D {
  Color color;
  float x, y, z;
};

template struct DesigInit0<ColorPoint3D, int, double>;
template struct DesigInit0<Color, int, double>; // expected-note{{instantiation}}

template<typename T, int Subscript1, int Subscript2,
         typename Val1, typename Val2>
struct DesigArrayInit0 {
  void f(Val1 val1, Val2 val2) {
    T array = {
      [Subscript1] = val1,
      [Subscript2] = val2 // expected-error{{exceeds array bounds}}
    };

    int array2[10] = { [5] = 3 };
  }
};

template struct DesigArrayInit0<int[8], 5, 3, float, int>;
template struct DesigArrayInit0<int[8], 5, 13, float, int>; // expected-note{{instantiation}}

template<typename T, int Subscript1, int Subscript2,
         typename Val1>
struct DesigArrayRangeInit0 {
  void f(Val1 val1) {
    T array = {
      [Subscript1...Subscript2] = val1 // expected-error{{exceeds}}
    };
  }
};

template struct DesigArrayRangeInit0<int[8], 3, 5, float>;
template struct DesigArrayRangeInit0<int[8], 5, 13, float>; // expected-note{{instantiation}}

// ---------------------------------------------------------------------
// Compound literals
// ---------------------------------------------------------------------
template<typename T, typename Arg1, typename Arg2>
struct CompoundLiteral0 {
  T f(Arg1 a1, Arg2 a2) {
    return (T){a1, a2};
  }
};

template struct CompoundLiteral0<Point2D, int, float>;
