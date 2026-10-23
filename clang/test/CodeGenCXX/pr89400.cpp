template <typename T>
struct foo {
  template <typename U>
  friend foo<U> make_foo() {
    return foo<U>{};
  }
};

template <typename U>
foo<U> make_foo();

void f() {
  make_foo<int>();
}