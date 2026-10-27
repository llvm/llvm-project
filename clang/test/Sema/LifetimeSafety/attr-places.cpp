// RUN: %clang_cc1 -fsyntax-only -verify %s

// Sema checks for the place attributes:
//   [[clang::places(...)]], [[clang::place(p)]],
//   [[clang::lifetimebound(p)]], [[clang::invalidates(p)]].

struct [[clang::places(keys, values)]] Table {
  [[clang::place(keys)]] int keys_[4];
  [[clang::place(values)]] int values_[4];

  const int *Key(int i) const [[clang::lifetimebound(keys)]];
  const int *Value(int i) const [[clang::lifetimebound(values)]];
  const int *Any() const [[clang::lifetimebound(interior)]];
  const Table &Self() const [[clang::lifetimebound]];

  void Intern() [[clang::invalidates(keys)]];
  void Both() [[clang::invalidates(keys, values)]];
  void Clear() [[clang::invalidates(interior)]];
  void Destroy() [[clang::invalidates]];

  // Errors.
  const int *Typo() const [[clang::lifetimebound(valeus)]]; // expected-error {{unknown place 'valeus' in 'Table'}}
  void Typo2() [[clang::invalidates(keyz)]];               // expected-error {{unknown place 'keyz' in 'Table'}}
  void Mixed() [[clang::invalidates(interior, keys)]];     // expected-error {{'interior' cannot be combined with named places in 'clang::invalidates'}}
  void Dup() [[clang::invalidates(keys, keys)]];           // expected-error {{duplicate place 'keys'}}
  [[clang::place(nope)]] int nope_;                        // expected-error {{unknown place 'nope' in 'Table'}}
  [[clang::place(interior)]] int interior_;                // expected-error {{'interior' is a reserved place name}}
  [[clang::place("keys")]] int str_;                       // expected-error {{'clang::place' attribute requires an identifier}}
};
// expected-note@7 4 {{places of 'Table' are declared here}}

// Out-of-line definitions resolve places against the class.
const int *Table::Key(int i) const [[clang::lifetimebound(keys)]] { return &keys_[i]; }
void Table::Intern() [[clang::invalidates(keys)]] {}

struct NoPlaces {
  void f() [[clang::invalidates(keys)]];      // expected-error {{'NoPlaces' declares no places; use '[[clang::places(...)]]' on the class}}
  const int *g() [[clang::lifetimebound(k)]]; // expected-error {{'NoPlaces' declares no places; use '[[clang::places(...)]]' on the class}}
  [[clang::place(keys)]] int x;               // expected-error {{'NoPlaces' declares no places; use '[[clang::places(...)]]' on the class}}
  void ok() [[clang::invalidates(interior)]]; // OK: every class has an interior.
  const int *ok2() [[clang::lifetimebound(interior)]];
};

struct [[clang::places(interior)]] Reserved {};     // expected-error {{'interior' is a reserved place name}}
struct [[clang::places(a, a)]] Duplicate {};        // expected-error {{duplicate place 'a'}}
struct [[clang::places]] Empty {};                  // expected-error {{'clang::places' attribute takes at least 1 argument}}
struct [[clang::places("a")]] Str {};               // expected-error {{'clang::places' attribute requires an identifier}}
[[clang::places(a)]] void NotAClass();              // expected-error {{'clang::places' attribute only applies to classes}}
[[clang::place(a)]] int NotAField;                  // expected-error {{'clang::place' attribute only applies to non-static data members}}

// On parameters.
const int *P1(Table &t [[clang::lifetimebound(keys)]]);
void P2(Table *t [[clang::invalidates(values)]]);
void P3(Table &t [[clang::invalidates]]);
void P4(Table &t [[clang::invalidates(interior)]]);
const int *P5(Table &t [[clang::lifetimebound(nope)]]);  // expected-error {{unknown place 'nope' in 'Table'}}
void P6(int *p [[clang::invalidates]]);                  // OK: the object `*p` is freed.
void P7(int *p [[clang::invalidates(interior)]]);        // expected-error {{'invalidates(interior)' requires a parameter of class type (by pointer or reference)}}
int *P8(int *p [[clang::lifetimebound(interior)]]);      // expected-error {{'lifetimebound(interior)' requires a parameter of class type (by pointer or reference)}}
void P9(Table t [[clang::invalidates]]);                 // expected-error {{'invalidates' attribute only applies to pointer or reference parameters and the implicit object parameter}}
void P10(NoPlaces &n [[clang::invalidates(keys)]]);      // expected-error {{'NoPlaces' declares no places; use '[[clang::places(...)]]' on the class}}
[[clang::invalidates]] void P11();                       // expected-error {{'clang::invalidates' attribute only applies to parameters and implicit object parameters}}
int P12 [[clang::invalidates]];                          // expected-error {{'clang::invalidates' attribute only applies to parameters and implicit object parameters}}
void P13(int i [[clang::invalidates]]);                  // expected-error {{'invalidates' attribute only applies to pointer or reference parameters and the implicit object parameter}}
void P14() [[clang::invalidates]];                       // expected-error {{'invalidates' attribute cannot be applied; non-member function has no implicit object parameter}}
struct S15 {
  static void f() [[clang::invalidates(interior)]];      // expected-error {{'invalidates' attribute cannot be applied; static member function has no implicit object parameter}}
  S15() [[clang::invalidates]];                          // OK: constructors may invalidate (e.g. a move constructor's source).
  ~S15() [[clang::invalidates]];
};

// Dependent types are checked on instantiation, not at the template.
template <typename T> const T *D1(T &t [[clang::lifetimebound(keys)]]);
template <typename T> void D2(T *t [[clang::invalidates(whatever)]]);
template <typename T> struct [[clang::places(items)]] Templ {
  [[clang::place(items)]] T items_[4];
  const T *Item(int i) const [[clang::lifetimebound(items)]];
  void Add() [[clang::invalidates(items)]];
};
Templ<int> ti;

// Forward-declared classes cannot be checked yet.
struct Later;
void F1(Later &l [[clang::invalidates(anything)]]);
struct [[clang::places(anything)]] Later {};
