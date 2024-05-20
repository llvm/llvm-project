// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

auto x0 = requires (this int) { true; }; // expected-error {{a requires expression cannot have an explicit object parameter}}
auto x1 = requires (int, this int) { true; }; // expected-error {{a requires expression cannot have an explicit object parameter}}

template<this auto> // expected-error {{expected template parameter}}
void f(); // expected-error {{no function template matches function template specialization 'f'}}

struct A {
  template<typename T>
  void f0(this T); // expected-note 2{{attempt to specialize declaration here}}

  template<>
  void f0(this short);

  template<>
  void f0(long); // expected-error {{an explicit specialization of an explicit object member function must have an explicit object parameter}}

  template<typename T>
  void g0(T); // expected-note 2{{attempt to specialize declaration here}}

  template<>
  void g0(short);

  template<>
  void g0(this long); // expected-error {{an explicit specialization of an implicit object member function cannot have an explicit object parameter}}

  template<typename T>
  static void h0(T); // expected-note 2{{attempt to specialize declaration here}}

  template<>
  void h0(short);

  template<>
  void h0(this long); // expected-error {{an explicit specialization of a static member function cannot have an explicit object parameter}}
};

template<>
void A::f0(this signed);

template<>
void A::f0(unsigned); // expected-error {{an explicit specialization of an explicit object member function must have an explicit object parameter}}

template<>
void A::g0(signed);

template<>
void A::g0(this unsigned); // expected-error {{an explicit specialization of an implicit object member function cannot have an explicit object parameter}}

template<>
void A::h0(signed);

template<>
void A::h0(this unsigned); // expected-error {{an explicit specialization of a static member function cannot have an explicit object parameter}}

template<typename T>
struct B {
  void f1(this int); // expected-note {{member declaration nearly matches}}

  void g1(int); // expected-note {{member declaration nearly matches}}

  static void h1(int); // expected-note {{member declaration nearly matches}}
};

template<>
void B<short>::f1(this int);

template<>
void B<long>::f1(int); // expected-error {{out-of-line declaration of 'f1' does not match any declaration in 'B<long>'}}

template<>
void B<short>::g1(int);

template<>
void B<long>::g1(this int); // expected-error {{out-of-line declaration of 'g1' does not match any declaration in 'B<long>'}}

template<>
void B<short>::h1(int);

template<>
void B<long>::h1(this int); // expected-error {{out-of-line declaration of 'h1' does not match any declaration in 'B<long>'}}

template<typename T>
struct C {
  template<typename U>
  void f2(this U); // expected-note {{attempt to specialize declaration here}}

  template<>
  void f2(this short);

  template<>
  void f2(long); // expected-error {{an explicit specialization of an explicit object member function must have an explicit object parameter}}

  template<typename U>
  void g2(U); // expected-note {{attempt to specialize declaration here}}

  template<>
  void g2(short);

  template<>
  void g2(this long); // expected-error {{an explicit specialization of an implicit object member function cannot have an explicit object parameter}}

  template<typename U>
  static void h2(U); // expected-note {{attempt to specialize declaration here}}

  template<>
  void h2(short);

  template<>
  void h2(this long); // expected-error {{an explicit specialization of a static member function cannot have an explicit object parameter}}
};

template struct C<int>; // expected-note {{in instantiation of}}

template<typename T>
struct D {
  template<typename U>
  void f3(this U);

  template<typename U>
  void g3(U);

  template<typename U>
  static void h3(U);
};

template<>
template<typename U>
void D<short>::f3(this U);

template<>
template<typename U>
void D<long>::f3(U); // expected-error {{out-of-line declaration of 'f3' does not match any declaration in 'D<long>'}}

template<>
template<typename U>
void D<short>::g3(U);

template<>
template<typename U>
void D<long>::g3(this U); // expected-error {{out-of-line declaration of 'g3' does not match any declaration in 'D<long>'}}

template<>
template<typename U>
void D<short>::h3(U);

template<>
template<typename U>
void D<long>::h3(this U); // expected-error {{out-of-line declaration of 'h3' does not match any declaration in 'D<long>'}}
