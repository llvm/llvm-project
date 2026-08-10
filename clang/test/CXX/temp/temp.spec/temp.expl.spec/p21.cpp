// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X {
  void mf1(T);
  template<typename U> void mf2(T, U); // expected-note{{previous}}
};

template<>
void X<int>::mf1(int i = 17) // expected-error{{default}}
{
}

template<> template<>
void X<int>::mf2(int, int = 17) // expected-error{{default}}
{ }

template<> template<typename U> 
void X<int>::mf2(int, U = U()) // expected-error{{default}}
{
}

template<>
struct X<float> {
  void mf1(float);
};

void X<float>::mf1(float = 3.14f)  // okay
{
}
