// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

using size_t = decltype(sizeof(int));
void *operator new(size_t, void *p) { return p; }

struct myfunction {
  union storage_t {
    char buffer[100];
    size_t max_align;
  } storage;

  template <typename Func> myfunction(Func fn) {
    new (&storage.buffer) Func(fn);
  }
  void operator()();
};

myfunction create_func() {
  int n;
  auto c = [&n] {};
  return c; // expected-warning {{Address of stack memory associated with local variable 'n' is still referred to by a temporary object on the stack upon returning to the caller.  This will be a dangling reference}}
}
void gh_66221() {
  create_func()();
}
