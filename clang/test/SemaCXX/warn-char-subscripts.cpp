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

void t3() {
  int array[50] = { 0 };
  int val = array[' ']; // no warning, subscript is a literal
}
void t4() {
  int array[50] = { 0 };
  int val = '('[array]; // no warning, subscript is a literal
}
void t5() {
  int array[50] = { 0 };
  int val = array['\x11']; // no warning, subscript is a literal
}

void test() {
  t1<char>(); // expected-note {{in instantiation of function template specialization 't1<char>' requested here}}
  t2<char>(); // expected-note {{in instantiation of function template specialization 't2<char>' requested here}}
}

