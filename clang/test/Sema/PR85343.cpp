// RUN: %clang_cc1 -std=c++14 -verify %s
// expected-no-diagnostics

template <typename c> auto ab() -> c ;

template <typename> struct e {};

template <typename f> struct ac {
  template <typename h> static e<decltype(ab<h>()(ab<int>))> i;
  decltype(i<f>) j;
};

struct d {
  template <typename f>
  d(f) { 
    ac<f> a;
  }
};
struct a {
  d b = [=](auto) { (void)[this] {}; };
};
void b() { new a; }
