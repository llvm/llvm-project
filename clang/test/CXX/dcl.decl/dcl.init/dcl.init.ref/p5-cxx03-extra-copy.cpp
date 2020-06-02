// RUN: %clang_cc1 -fsyntax-only -std=c++03 -Wbind-to-temporary-copy -verify %s

// C++03 requires that we check for a copy constructor when binding a
// reference to a temporary, since we are allowed to make a copy, Even
// though we don't actually make that copy, make sure that we diagnose
// cases where that copy constructor is somehow unavailable.  As an
// extension, this is only a warning.

struct X1 {
  X1();
  explicit X1(const X1&);
};

struct X2 {
  X2();

private:
  X2(const X2&); // expected-note{{declared private here}}
};

struct X3 {
  X3(); // expected-note{{requires 0 arguments, but 1 was provided}}

private:
  X3(X3&); // expected-note{{candidate constructor not viable: expects an l-value for 1st argument}}
};

// Check for instantiation of default arguments
template<typename T>
T get_value_badly() {
  double *dp = 0;
  // The extension doesn't extend far enough to turn this error into a warning.
  T *tp = dp; // expected-error{{cannot initialize a variable of type 'int *' with an lvalue of type 'double *'}}
  return T();
}

template<typename T>
struct X4 {
  X4();
  X4(const X4&, T = get_value_badly<T>()); // expected-note{{in instantiation of}}
}; 

// Check for "dangerous" default arguments that could cause recursion.
struct X5 {
  X5();
  X5(const X5&, const X5& = X5()); // expected-error {{recursive evaluation of default argument}} expected-note {{used here}}
};

void g1(const X1&);
void g2(const X2&);
void g3(const X3&);
void g4(const X4<int>&);
void g5(const X5&);

void test() {
  g1(X1());
  g2(X2()); // expected-warning{{C++98 requires an accessible copy constructor for class 'X2' when binding a reference to a temporary; was private}}
  g3(X3()); // expected-warning{{no viable constructor copying parameter of type 'X3'}}
  g4(X4<int>());
  g5(X5());
}

// Check that unavailable copy constructors still cause SFINAE failures.
template<int> struct int_c { };

template<typename T> T f(const T&);

// Would be ambiguous with the next g(), except the instantiation failure in
// sizeof() prevents that.
template<typename T>
int &g(int_c<sizeof(f(T()))> * = 0);

template<typename T> float &g();

void h() {
  float &fp2 = g<X3>();  // Not ambiguous.
}
