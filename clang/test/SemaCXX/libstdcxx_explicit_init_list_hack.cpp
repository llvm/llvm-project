// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Wsystem-headers %s

// libstdc++4.6 in debug mode has explicit default constructors.
// stlport has this for all containers.
#ifdef BE_THE_HEADER
#pragma clang system_header
namespace std {
namespace __debug {
template <class T>
class vector {
public:
  explicit vector() {} // expected-warning 2 {{should not be explicit}}
};
}
}
#else

#define BE_THE_HEADER
#include __FILE__

struct { int a, b; std::__debug::vector<int> c; } e[] = { {1, 1} }; // expected-note{{used in initialization here}}
// expected-warning@+1 {{expression with side effects has no effect in an unevaluated context}}
decltype(new std::__debug::vector<int>[1]{}) x; // expected-note{{used in initialization here}}
#endif
