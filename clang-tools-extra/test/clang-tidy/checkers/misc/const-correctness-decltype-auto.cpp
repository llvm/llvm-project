// RUN: %check_clang_tidy -std=c++14-or-later %s misc-const-correctness %t \
// RUN: -config='{CheckOptions: {misc-const-correctness.WarnPointersAsValues: true}}' \
// RUN: -- -fno-delayed-template-parsing

// 'decltype(auto)' cannot be combined with 'const', so variables declared with
// it must not be diagnosed, whatever type they deduce to.

int global = 0;
int &get_ref() { return global; }
int *get_ptr() { return &global; }

void sink(int);

void decltype_auto_is_ignored() {
  int i = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'i' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const i = 42;

  decltype(auto) value = i;
  decltype(auto) ref = get_ref();
  decltype(auto) ptr = get_ptr();
  decltype(auto) (paren) = 42;
  sink(value);
  sink(ref);
  sink(*ptr);
  sink(paren);
}

template <typename T>
void decltype_auto_in_template(T t) {
  decltype(auto) value = t;
  sink(value);
}
void instantiate_template() { decltype_auto_in_template(0); }
