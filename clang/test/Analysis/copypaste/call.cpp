// RUN: %clang_cc1 -analyze -std=c++1z -analyzer-checker=alpha.clone.CloneChecker -verify %s

// expected-no-diagnostics

bool a();
bool b();

// Calls method a with some extra code to pass the minimum complexity
bool foo1(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return a();
  return true;
}

// Calls method b with some extra code to pass the minimum complexity
bool foo2(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return b();
  return true;
}

// Test that we don't crash on function pointer calls

bool (*funcPtr)(int);

bool fooPtr1(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return funcPtr(1);
  return true;
}

// Test that we respect the template arguments of function templates

template<typename T, unsigned N>
bool templateFunc() { unsigned i = N; return false; }

bool fooTemplate1(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return templateFunc<int, 1>();
  return true;
}

bool fooTemplate2(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return templateFunc<long, 1>();
  return true;
}

bool fooTemplate3(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return templateFunc<long, 2>();
  return true;
}

// Test that we don't just concatenate the template arguments into a string
// without having any padding between them (e.g. foo<X, XX>() != foo<XX, X>()).

class X {};
class XX {};

template<typename T1, typename T2>
bool templatePaddingFunc() { return false; }

bool fooTemplatePadding1(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return templatePaddingFunc<X, XX>();
  return true;
}

bool fooTemplatePadding2(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    return templatePaddingFunc<XX, X>();
  return true;
}

// Test that we don't crash on member functions of template instantiations.

template<typename T>
struct A {
  void foo(T t) {}
};

void fooTestInstantiation() {
  A<int> a;
  a.foo(1);
}
