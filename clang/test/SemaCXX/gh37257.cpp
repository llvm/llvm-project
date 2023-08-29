// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class T>
T&& create();

template<class T, class... Args>
void test() {
  T t(create<Args>()...); // expected-error{{variable has incomplete type 'int[]'}}
  (void) t;
}

struct A;

int main() {
  test<int[]>(); // expected-note {{in instantiation of function template specialization 'test<int[]>' requested here}}
}
