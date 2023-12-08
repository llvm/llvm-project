// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename> struct PassRefPtr { };
template<typename T> struct RefPtr {
  RefPtr& operator=(const RefPtr&) { int a[sizeof(T) ? -1 : -1];} // expected-error 2 {{array with a negative size}}
  RefPtr& operator=(const PassRefPtr<T>&);
};

struct A { RefPtr<int> a; };  // expected-note {{instantiation of member function 'RefPtr<int>::operator=' requested here}}
struct B : RefPtr<float> { }; // expected-note {{in instantiation of member function 'RefPtr<float>::operator=' requested here}}

void f() {
  A a1, a2;
  a1 = a2;

  B b1, b2;
  b1 = b2;
}
