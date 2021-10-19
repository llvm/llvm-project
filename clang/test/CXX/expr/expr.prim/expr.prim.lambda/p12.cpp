// RUN: %clang_cc1 -std=c++11 %s -Wunused -Wno-unused-but-set-variable -Wno-unused-lambda-capture -verify

void odr_used() {
  int i = 17;
  [i]{}();
}

struct ReachingThis {
  static void static_foo() {
    (void)[this](){}; // expected-error{{'this' cannot be captured in this context}}

    struct Local {
      int i;

      void bar() {
        (void)[this](){};
        (void)[&](){i = 7; };
      }
    };
  }

  void foo() {
    (void)[this](){};
    
    struct Local {
      int i;

      static void static_bar() {
        (void)[this](){}; // expected-error{{'this' cannot be captured in this context}}
        (void)[&](){i = 7; }; // expected-error{{invalid use of member 'i' in static member function}}
      }
    };
  }
};

void immediately_enclosing(int i) { // expected-note{{'i' declared here}}
  [i]() {
    [i] {}();
  }();

  [=]() {
    [i] {}();
  }();

  []() {      // expected-note{{lambda expression begins here}} expected-note 2 {{capture 'i' by}} expected-note 2 {{default capture by}}
    [i] {}(); // expected-error{{variable 'i' cannot be implicitly captured in a lambda with no capture-default specified}}
  }();
}

void f1(int i) { // expected-note{{declared here}}
  int const N = 20;
  auto m1 = [=]{
    int const M = 30;
    auto m2 = [i]{
      int x[N][M];
      x[0][0] = i;
    }; 
    (void)N;
    (void)M;
    (void)m2;
  };
  struct s1 {
    int f;
    void work(int n) { // expected-note{{declared here}}
      int m = n*n;
      int j = 40; // expected-note{{declared here}}
      auto m3 = [this, m] { // expected-note 3{{lambda expression begins here}} expected-note 2 {{capture 'i' by}} expected-note 2 {{capture 'j' by}} expected-note 2 {{capture 'n' by}}
        auto m4 = [&,j] { // expected-error{{variable 'j' cannot be implicitly captured in a lambda with no capture-default specified}}
          int x = n; // expected-error{{variable 'n' cannot be implicitly captured in a lambda with no capture-default specified}}
          x += m;
          x += i; // expected-error{{variable 'i' cannot be implicitly captured in a lambda with no capture-default specified}}
          x += f;
        };
      };
    } 
  };
}
