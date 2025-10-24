// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -Wuninitialized-const-pointer -verify %s

template <class T>
void ignore_template(const T *) {}
void ignore(const int *) {}
void dont_ignore_non_empty(const int *) { ; } 
void dont_ignore_block(const int *) { {} }
void dont_ignore_try_block(const int *) try {
} catch (...) {
}
int const_ptr_use(const int *);

void f(int a) {
  int i;
  const_ptr_use(&i);             // expected-warning {{variable 'i' is uninitialized when passed as a const pointer argument here}}
  int j = j + const_ptr_use(&j); // expected-warning {{variable 'j' is uninitialized when used within its own initialization}}
  int k = k;                     // expected-warning {{variable 'k' is uninitialized when used within its own initialization}}
  const_ptr_use(&k);

  // Only report if a variable is always uninitialized at the point of use
  int l;
  if (a < 42)
    l = 1;
  const_ptr_use(&l);

  // Don't report if the called function is known to be empty.
  int m;
  ignore_template(&m);
  ignore(&m);
  dont_ignore_non_empty(&m); // expected-warning {{variable 'm' is uninitialized when passed as a const pointer argument here}}
  int n;
  dont_ignore_block(&n); // expected-warning {{variable 'n' is uninitialized when passed as a const pointer argument here}}
  int o;
  dont_ignore_try_block(&o); // expected-warning {{variable 'o' is uninitialized when passed as a const pointer argument here}}
}
