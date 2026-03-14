// Header for PCH test cxx-for-range.cpp

struct S {
  int *begin();
  int *end();
};

struct T { };
char *begin(T);
char *end(T);

namespace NS {
  struct U { };
  char *begin(U);
  char *end(U);
}
using NS::U;

void f() {
  char a[3] = { 0, 1, 2 };
  for (auto w : a)
    for (auto x : S())
      for (auto y : T())
        for (auto z : U())
          ;
}

template<typename A>
void g() {
  A a[3] = { 0, 1, 2 };
  for (auto &v : a)
    for (auto x : S())
      for (auto y : T())
        for (auto z : U())
          ;
}
