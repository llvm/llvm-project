// RUN: %clang_cc1 -std=c++14 -verify=both,expected %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++14 -verify=both,ref      %s



constexpr int(*null_ptr)() = nullptr;
constexpr int test4 = (*null_ptr)(); // both-error {{must be initialized by a constant expression}} \
                                     // both-note {{evaluates to a null function pointer}}

struct E {
  int n = 0;
  struct {
    void *x = this;
  };
  void *y = this;
};
constexpr E e1 = E();
static_assert(e1.x != e1.y, "");
constexpr E e2 = E{0};
static_assert(e2.x != e2.y, "");

struct S {
  int &&a = 2;
  int b[1]{a};
};
constexpr int foo() {
  S s{12};
  return s.b[0];
}
static_assert(foo() == 12, "");

int arr[3]; // both-note {{declared here}}
constexpr bool f() { // both-error {{constexpr function never produces a constant expression}}
  int &r  = arr[3]; // both-note {{read of dereferenced one-past-the-end pointer}} \
                    // both-warning {{array index 3 is past the end of the array}}
  return true;
}
