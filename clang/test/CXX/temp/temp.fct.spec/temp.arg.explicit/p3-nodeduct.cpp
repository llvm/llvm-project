// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5811
template <class F> void Call(F f) { f(1); }
template <typename T> void f(T);
void a() { Call(f<int>); }

// Check the conversion of a template-id to a pointer
template<typename T, T* Address> struct Constant { };
Constant<void(int), &f<int> > constant0;

template<typename T, T* Address> void constant_func();
void test_constant_func() {
  constant_func<void(int), &f<int> >();
}


// Check typeof() on a template-id referring to a single function
template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

int typeof0[is_same<__typeof__(f<int>), void (int)>::value? 1 : -1];
int typeof1[is_same<__typeof__(&f<int>), void (*)(int)>::value? 1 : -1];

template <typename T> void g(T); // expected-note{{possible target for call}}
template <typename T> void g(T, T); // expected-note{{possible target for call}}

int typeof2[is_same<__typeof__(g<float>), void (int)>::value? 1 : -1]; // \
     // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
