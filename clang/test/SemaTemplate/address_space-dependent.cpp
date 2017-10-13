// RUN: %clang_cc1 -x c++ -std=c++14 -fsyntax-only -verify %s

template <int I, int J, int K>
void car() {
  int __attribute__((address_space(I))) __attribute__((address_space(J))) * Y;  // expected-error {{multiple address spaces specified for type}}
  int *__attribute__((address_space(I))) __attribute__((address_space(J))) * Z; // expected-error {{multiple address spaces specified for type}}

  __attribute__((address_space(I))) int local;        // expected-error {{automatic variable qualified with an address space}}
  __attribute__((address_space(J))) int array[5];     // expected-error {{automatic variable qualified with an address space}}
  __attribute__((address_space(I))) int arrarr[5][5]; // expected-error {{automatic variable qualified with an address space}}

  __attribute__((address_space(J))) * x; // expected-error {{C++ requires a type specifier for all declarations}}

  __attribute__((address_space(I))) float *B;

  typedef __attribute__((address_space(J))) int AS2Int;
  struct HasASFields {
    AS2Int typedef_as_field; // expected-error {{field may not be qualified with an address space}}
  };

  struct _st {
    int x, y;
  } s __attribute((address_space(I))) = {1, 1};
}

template <int I>
struct HasASTemplateFields {
  __attribute__((address_space(I))) int as_field; // expected-error {{field may not be qualified with an address space}}
};

template <int I, int J>
void foo(__attribute__((address_space(I))) float *a, // expected-note {{candidate template ignored: substitution failure [with I = 1, J = 2]: parameter may not be qualified with an address space}}
         __attribute__((address_space(J))) float b) {
  *a = 5.0f + b;
}

template void foo<1, 2>(float *, float); // expected-error {{explicit instantiation of 'foo' does not refer to a function template, variable template, member function, member class, or static data member}}

template <int I>
void neg() {
  __attribute__((address_space(I))) int *bounds; // expected-error {{address space is negative}}
}

template <long int I>
void tooBig() {
  __attribute__((address_space(I))) int *bounds; // expected-error {{address space is larger than the maximum supported (8388598)}}
}

template <long int I>
void correct() {
  __attribute__((address_space(I))) int *bounds;
}

template <int I, int J>
char *cmp(__attribute__((address_space(I))) char *x, __attribute__((address_space(J))) char *y) {
  return x < y ? x : y; // expected-error {{comparison of distinct pointer types ('__attribute__((address_space(1))) char *' and '__attribute__((address_space(2))) char *')}}
}

typedef void ft(void);

template <int I>
struct fooFunction {
  __attribute__((address_space(I))) void **const base = 0;

  void *get_0(void) {
    return base[0]; // expected-error {{cannot initialize return object of type 'void *' with an lvalue of type '__attribute__((address_space(1))) void *}}
  }

  __attribute__((address_space(I))) ft qf; // expected-error {{function type may not be qualified with an address space}}
  __attribute__((address_space(I))) char *test3_val;

  void test3(void) {
    extern void test3_helper(char *p); // expected-note {{passing argument to parameter 'p' here}}
    test3_helper(test3_val);           // expected-error {{cannot initialize a parameter of type 'char *' with an lvalue of type '__attribute__((address_space(1))) char *'}}
  }
};

template <typename T, int N>
int GetAddressSpaceValue(T __attribute__((address_space(N))) * p) {
  return N;
}

template <unsigned A> int __attribute__((address_space(A))) *same_template();
template <unsigned B> int __attribute__((address_space(B))) *same_template();
void test_same_template() { (void) same_template<0>(); }

template <unsigned A> int __attribute__((address_space(A))) *different_template(); // expected-note {{candidate function [with A = 0]}}
template <unsigned B> int __attribute__((address_space(B+1))) *different_template(); // expected-note {{candidate function [with B = 0]}}
void test_different_template() { (void) different_template<0>(); } // expected-error {{call to 'different_template' is ambiguous}}

template <typename T> struct partial_spec_deduce_as;
template <typename T, unsigned AS> 
struct partial_spec_deduce_as <__attribute__((address_space(AS))) T *> {
   static const unsigned value = AS;  
}; 

int main() {  
  int __attribute__((address_space(1))) * p1;
  int p = GetAddressSpaceValue(p1);

  car<1, 2, 3>(); // expected-note {{in instantiation of function template specialization 'car<1, 2, 3>' requested here}}
  HasASTemplateFields<1> HASTF;
  neg<-1>(); // expected-note {{in instantiation of function template specialization 'neg<-1>' requested here}}
  correct<0x7FFFF6>();
  tooBig<8388650>(); // expected-note {{in instantiation of function template specialization 'tooBig<8388650>' requested here}}

  __attribute__((address_space(1))) char *x;
  __attribute__((address_space(2))) char *y;
  cmp<1, 2>(x, y); // expected-note {{in instantiation of function template specialization 'cmp<1, 2>' requested here}}

  fooFunction<1> ff;
  ff.get_0(); // expected-note {{in instantiation of member function 'fooFunction<1>::get_0' requested here}}
  ff.qf();
  ff.test3(); // expected-note {{in instantiation of member function 'fooFunction<1>::test3' requested here}}
  
  static_assert(partial_spec_deduce_as<int __attribute__((address_space(3))) *>::value == 3, "address space value has been incorrectly deduced"); 

  return 0;
}
