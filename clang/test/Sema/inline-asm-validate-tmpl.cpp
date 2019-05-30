// RUN: %clang_cc1 -triple i686 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64 -fsyntax-only -verify %s


// this template, when instantiated with 300, violates the range contraint
template <int N> void	test(int value)
{
  asm("rol %1, %0" :"=r"(value): "I"(N + 1)); // expected-error{{value '301' out of range for constraint 'I'}}
}

int		main() { test<300>(10); } // expected-note{{in instantiation of function template specialization 'test<300>' requested here}}


// this template is not used, but the error is detectable
template <int N> void	testb(int value)
{
   asm("rol %1, %0" :"=r"(value): "I"(301)); // expected-error{{value '301' out of range for constraint 'I'}}
}

// these should compile without error
template <int N> void	testc(int value)
{
	asm("rol %1, %0" :"=r"(value): "I"(N + 1));
}
int	foo() { testc<2>(10); }
