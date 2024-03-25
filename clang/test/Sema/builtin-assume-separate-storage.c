// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

void *nonconst(void);

void test1(int *a, int *b) {
  __builtin_assume_separate_storage(a, b);
  // Separate storage assumptions evaluate their arguments unconditionally, like
  // assume_aligned but *unlike* assume. Check that we don't warn on it.
  __builtin_assume_separate_storage(a, nonconst());
  __builtin_assume_separate_storage(nonconst(), a);
  __builtin_assume_separate_storage(a, 3); // expected-error {{incompatible integer to pointer conversion}}
  __builtin_assume_separate_storage(3, a); // expected-error {{incompatible integer to pointer conversion}}
}
