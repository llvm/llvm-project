// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Winitializer-overrides %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Woverride-init %s

template <typename T> struct Foo {
  struct SubFoo {
    int bar1;
    int bar2;
  };

  static void Test() { SubFoo sf = {.bar1 = 10, .bar2 = 20}; } // Expected no warning
};

void foo() {
  Foo<int>::Test();
  Foo<bool>::Test();
  Foo<float>::Test();
}

template <typename T> struct Bar {
  struct SubFoo {
    int bar1;
    int bar2;
  };

  static void Test() { SubFoo sf = {.bar1 = 10,    // expected-note 2 {{previous initialization is here}}
                                    .bar1 = 20}; } // expected-warning 2 {{initializer overrides prior initialization of this subobject}}
};

void bar() {
  Bar<int>::Test();  // expected-note {{in instantiation of member function 'Bar<int>::Test' requested here}}
  Bar<bool>::Test(); // expected-note {{in instantiation of member function 'Bar<bool>::Test' requested here}}
}
