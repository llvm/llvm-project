// RUN: %clang_cc1 -std=c++11 %s -Wunused -Wno-unused-lambda-capture -verify


struct X {
  X(const X&) = delete; // expected-note 2{{explicitly marked deleted}}
  X(X&);
};

void test_capture(X x) {
  [x] { }(); // okay: non-const copy ctor

  [x] {
    [x] { // expected-error{{call to deleted constructor of 'X'}}
    }();
  }();

  [x] {
    [&x] {
      [x] { // expected-error{{call to deleted constructor of 'const X'}}
      }();
    }();
  }();

  int a;
  [=] {
    [&] {
      int &x = a;  // expected-error{{binding reference of type 'int' to value of type 'const int' drops 'const' qualifier}}
      int &x2 = a; // expected-error{{binding reference of type 'int' to value of type 'const int' drops 'const' qualifier}}
    }();
  }();

  [=] {
    [&a] {
      [&] {
        int &x = a;  // expected-error{{binding reference of type 'int' to value of type 'const int' drops 'const' qualifier}}
        int &x2 = a; // expected-error{{binding reference of type 'int' to value of type 'const int' drops 'const' qualifier}}
      }();
    }();
  }();
}
