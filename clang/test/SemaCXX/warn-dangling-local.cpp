// RUN: %clang_cc1 -verify -std=c++11 -Wdangling-assignment-gsl %s

using T = int[];

void f() {
  int *p = &(int&)(int&&)0; // expected-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  p =  &(int&)(int&&)0; // expected-warning {{object backing the pointer p will be destroyed at the end of the full-expression}}

  int *q = (int *const &)T{1, 2, 3}; // expected-warning {{temporary whose address is used as value of local variable 'q' will be destroyed at the end of the full-expression}}
  q = (int *const &)T{1, 2, 3}; // expected-warning {{object backing the pointer q will be destroyed at the end of the full-expression}}

  // FIXME: We don't warn here because the 'int*' temporary is not const, but
  // it also can't have actually changed since it was created, so we could
  // still warn.
  int *r = (int *&&)T{1, 2, 3};

  // FIXME: The wording of this warning is not quite right. There are two
  // temporaries here: an 'int* const' temporary that points to the array, and
  // is lifetime-extended, and an array temporary that the pointer temporary
  // points to, which doesn't live long enough.
  int *const &s = (int *const &)T{1, 2, 3}; // expected-warning {{temporary bound to local reference 's' will be destroyed at the end of the full-expression}}
}

// PR38355
void g() {
  const int a[] = {a[0]};
  const int b[] = {a[0]};
}

namespace std {
// std::basic_string has a hard-coded gsl::owner attr.
struct basic_string {
  const char* c_str();
};
}  // namespace std
void test(const char* a) {
  // verify we're emitting the `-Wdangling-assignment-gsl` warning.
  a = std::basic_string().c_str(); // expected-warning {{object backing the pointer a will be destroyed at the end of the full-expression}}
}
