// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace DeductionForInstantiation {
  template<unsigned I, typename ...Types>
  struct X { };

  template<typename ...Types>
  void f0(X<sizeof...(Types), Types&...>) { }

  // No explicitly-specified arguments
  template void f0(X<0>);
  template void f0(X<1, int&>);
  template void f0(X<2, int&, short&>);

  // One explicitly-specified argument
  template void f0<float>(X<1, float&>);
  template void f0<double>(X<1, double&>);

  // Two explicitly-specialized arguments
  template void f0<char, unsigned char>(X<2, char&, unsigned char&>);
  template void f0<signed char, char>(X<2, signed char&, char&>);

  // FIXME: Extension of explicitly-specified arguments
  //  template void f0<short, int>(X<3, short&, int&, long&>);
}

namespace DeductionWithConversion {
  template<char...> struct char_values {
    static const unsigned value = 0;
  };

  template<int C1, char C3>
  struct char_values<C1, 12, C3> {
    static const unsigned value = 1;
  };

  int check0[char_values<1, 12, 3>::value == 1? 1 : -1];

  template<int...> struct int_values {
    static const unsigned value = 0;
  };

  template<unsigned char C1, unsigned char C3>
  struct int_values<C1, 12, C3> {
    static const unsigned value = 1;
  };

  int check1[int_values<256, 12, 3>::value == 0? 1 : -1];  
  int check2[int_values<3, 12, 3>::value == 1? 1 : -1];  
}
