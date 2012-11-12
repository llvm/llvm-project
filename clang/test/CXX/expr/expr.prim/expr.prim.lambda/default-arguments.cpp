// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -Wno-lambda-extensions -verify

void defargs() {
  auto l1 = [](int i, int j = 17, int k = 18) { return i + j + k; };
  int i1 = l1(1);
  int i2 = l1(1, 2);
  int i3 = l1(1, 2, 3);
}


void defargs_errors() {
  auto l1 = [](int i, 
               int j = 17, 
               int k) { }; // expected-error{{missing default argument on parameter 'k'}}

  auto l2 = [](int i, int j = i) {}; // expected-error{{default argument references parameter 'i'}}

  int foo;
  auto l3 = [](int i = foo) {}; // expected-error{{default argument references local variable 'foo' of enclosing function}}
}

struct NonPOD {
  NonPOD();
  NonPOD(const NonPOD&);
  ~NonPOD();
};

struct NoDefaultCtor {
  NoDefaultCtor(const NoDefaultCtor&); // expected-note{{candidate constructor}}
  ~NoDefaultCtor();
};

template<typename T>
void defargs_in_template_unused(T t) {
  auto l1 = [](const T& value = T()) { };
  l1(t);
}

template void defargs_in_template_unused(NonPOD);
template void defargs_in_template_unused(NoDefaultCtor);

template<typename T>
void defargs_in_template_used() {
  auto l1 = [](const T& value = T()) { }; // expected-error{{no matching constructor for initialization of 'NoDefaultCtor'}}
  l1(); // expected-note{{in instantiation of default function argument expression for 'operator()<NoDefaultCtor>' required here}}
}

template void defargs_in_template_used<NonPOD>();
template void defargs_in_template_used<NoDefaultCtor>(); // expected-note{{in instantiation of function template specialization}}

