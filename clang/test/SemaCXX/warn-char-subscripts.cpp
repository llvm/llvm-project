// RUN: %clang_cc1 -Wchar-subscripts -fsyntax-only -verify %s

template<typename T>
void t1() {
  int array[1] = { 0 };
  T subscript = 0;
  int val = array[subscript]; // expected-warning{{array subscript is of type 'char'}}
}

template<typename T>
void t2() {
  int array[1] = { 0 };
  T subscript = 0;
  int val = subscript[array]; // expected-warning{{array subscript is of type 'char'}}
}

void test() {
  t1<char>(); // expected-note {{in instantiation of function template specialization 't1<char>' requested here}}
  t2<char>(); // expected-note {{in instantiation of function template specialization 't2<char>' requested here}}
}

