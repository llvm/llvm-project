// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class T>
static int alpha(T c)
{
    return *c; // expected-error{{indirection requires pointer operand}}
}

template<class T>
static void
_shexp_match()
{
  switch(1) {
  case 1:
    alpha(1); // expected-note{{instantiation of function template}}
  }
}
int main() {
  _shexp_match<char>(); // expected-note{{instantiation of function template}}
  return 0;
}
