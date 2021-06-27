// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify -Wreserved-identifier %s

int foo__bar() { return 0; }    // expected-warning {{identifier 'foo__bar' is reserved because it contains '__'}}
static int _bar() { return 0; } // expected-warning {{identifier '_bar' is reserved because it starts with '_' at global scope}}
static int _Bar() { return 0; } // expected-warning {{identifier '_Bar' is reserved because it starts with '_' followed by a capital letter}}
int _barbouille() { return 0; } // expected-warning {{identifier '_barbouille' is reserved because it starts with '_' at global scope}}

void foo(unsigned int _Reserved) { // expected-warning {{identifier '_Reserved' is reserved because it starts with '_' followed by a capital letter}}
  unsigned int __1 =               // expected-warning {{identifier '__1' is reserved because it starts with '__'}}
      _Reserved;                   // no-warning
}

// This one is explicitly skipped by -Wreserved-identifier
void *_; // no-warning

template <class T> constexpr bool __toucan = true; // expected-warning {{identifier '__toucan' is reserved because it starts with '__'}}

template <class T>
concept _Barbotine = __toucan<T>; // expected-warning {{identifier '_Barbotine' is reserved because it starts with '_' followed by a capital letter}}

template <class __> // expected-warning {{'__' is reserved because it starts with '__'}}
struct BarbeNoire {};

template <class _not_reserved> // no-warning
struct BarbeJaune {};

template <class __> // expected-warning {{'__' is reserved because it starts with '__'}}
void BarbeRousse() {}

namespace _Barbidur { // expected-warning {{identifier '_Barbidur' is reserved because it starts with '_' followed by a capital letter}}

struct __barbidou {}; // expected-warning {{identifier '__barbidou' is reserved because it starts with '__'}}
struct _barbidou {};  // no-warning

int __barbouille; // expected-warning {{identifier '__barbouille' is reserved because it starts with '__'}}
int _barbouille;  // no-warning

int __babar() { return 0; } // expected-warning {{identifier '__babar' is reserved because it starts with '__'}}
int _babar() { return 0; }  // no-warning

} // namespace _Barbidur

class __barbapapa {     // expected-warning {{identifier '__barbapapa' is reserved because it starts with '__'}}
  void _barbabelle() {} // no-warning
  int _Barbalala;       // expected-warning {{identifier '_Barbalala' is reserved because it starts with '_' followed by a capital letter}}
};

enum class __menu { // expected-warning {{identifier '__menu' is reserved because it starts with '__'}}
  __some,           // expected-warning {{identifier '__some' is reserved because it starts with '__'}}
  _Other,           // expected-warning {{identifier '_Other' is reserved because it starts with '_' followed by a capital letter}}
  _other            // no-warning
};

enum _Menu { // expected-warning {{identifier '_Menu' is reserved because it starts with '_' followed by a capital letter}}
  _OtheR_,   // expected-warning {{identifier '_OtheR_' is reserved because it starts with '_' followed by a capital letter}}
  _other_    // expected-warning {{identifier '_other_' is reserved because it starts with '_' at global scope}}
};

enum {
  __some, // expected-warning {{identifier '__some' is reserved because it starts with '__'}}
  _Other, // expected-warning {{identifier '_Other' is reserved because it starts with '_' followed by a capital letter}}
  _other  // expected-warning {{identifier '_other' is reserved because it starts with '_' at global scope}}
};

static union {
  int _barbeFleurie; // no-warning
};

using _Barbamama = __barbapapa; // expected-warning {{identifier '_Barbamama' is reserved because it starts with '_' followed by a capital letter}}

int foobar() {
  return foo__bar(); // no-warning
}

namespace {
int _barbatruc; // no-warning
}

long double operator"" _BarbeBleue(long double) // expected-warning {{identifier '_BarbeBleue' is reserved because it starts with '_' followed by a capital letter}}
{
  return 0.;
}

long double operator""_SacreBleu(long double) // no-warning
{
  return 0.;
}

long double sacrebleu = operator"" _SacreBleu(1.2); // expected-warning {{identifier '_SacreBleu' is reserved because it starts with '_' followed by a capital letter}}
long double sangbleu = operator""_SacreBleu(1.2);   // no-warning

struct _BarbeRouge { // expected-warning {{identifier '_BarbeRouge' is reserved because it starts with '_' followed by a capital letter}}
} p;
struct _BarbeNoire { // expected-warning {{identifier '_BarbeNoire' is reserved because it starts with '_' followed by a capital letter}}
} * q;

struct Any {
  friend void _barbegrise(); // expected-warning {{identifier '_barbegrise' is reserved because it starts with '_' at global scope}}
};
