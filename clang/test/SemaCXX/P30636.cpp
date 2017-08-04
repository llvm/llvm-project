// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wimplicit-fallthrough %s
// expected-no-diagnostics

template<bool param>
int fallthrough_template(int i)
{
  switch (i) {
    case 1:
      if (param)
        return 3;
      [[clang::fallthrough]]; // no warning here, for an unreachable annotation (in the fallthrough_template<true> case) in a template function
    case 2:
      return 4;
    default:
      return 5;
  }
}
                                      
template int fallthrough_template<true>(int);

