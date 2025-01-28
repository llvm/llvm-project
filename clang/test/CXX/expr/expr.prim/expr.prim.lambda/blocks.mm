// RUN: %clang_cc1 -triple i686-pc-linux -std=c++11 -fblocks %s -verify

void block_capture_errors() {
  __block int var; // expected-note 2{{'var' declared here}}
  (void)[var] { }; // expected-error{{__block variable 'var' cannot be captured in a lambda}}

  (void)[=] { var = 17; }; // expected-error{{__block variable 'var' cannot be captured in a lambda}}
}

void conversion_to_block(int captured) {
  int (^b1)(int) = [=](int x) { return x + captured; };

  const auto lambda = [=](int x) { return x + captured; };
  int (^b2)(int) = lambda;
}

template<typename T>
class ConstCopyConstructorBoom {
public:
  ConstCopyConstructorBoom(ConstCopyConstructorBoom&);

  ConstCopyConstructorBoom(const ConstCopyConstructorBoom&) {
    T *ptr = 1; // expected-error{{cannot initialize a variable of type 'float *' with an rvalue of type 'int'}}
  }

  void foo() const;
};

void conversion_to_block_init(ConstCopyConstructorBoom<int> boom,
                              ConstCopyConstructorBoom<float> boom2) {
  const auto& lambda1([=] { boom.foo(); }); // okay

  const auto& lambda2([=] { boom2.foo(); }); // expected-note{{in instantiation of member function}}
  void (^block)(void) = lambda2;
}


void nesting() {
  int array[7]; // expected-note 2{{'array' declared here}}
  [=] () mutable {
    [&] {
      ^ {
        int i = array[2];
        i += array[3];
      }();
    }();
  }();

  [&] {
    [=] () mutable {
      ^ {
        int i = array[2]; // expected-error{{cannot refer to declaration with an array type inside block}}
        i += array[3];
      }();
    }();

    [=] () mutable {
      ^ {
        int i = 0;
        i += array[3]; // expected-error{{cannot refer to declaration with an array type inside block}}
      }();
    }();
  }();
}

namespace overloading {
  void bool_conversion() {
    if ([](){}) { // expected-warning{{address of lambda function pointer conversion operator will always evaluate to 'true'}}
    }

    bool b = []{}; // expected-warning{{address of lambda function pointer conversion operator will always evaluate to 'true'}}
    b = (bool)[]{};
  }

  void conversions() {
    int (*fp)(int) = [](int x) { return x + 1; };
    fp = [](int x) { return x + 1; };

    typedef int (*func_ptr)(int);
    fp = (func_ptr)[](int x) { return x + 1; };

    int (^bp)(int) = [](int x) { return x + 1; };
    bp = [](int x) { return x + 1; };

    typedef int (^block_ptr)(int);
    bp = (block_ptr)[](int x) { return x + 1; };
  }

  int &accept_lambda_conv(int (*fp)(int));
  float &accept_lambda_conv(int (^bp)(int));

  void call_with_lambda() {
    int &ir = accept_lambda_conv([](int x) { return x + 1; });
  }

  template<typename T> using id = T;

  auto a = [](){};
  struct C : decltype(a) {
    using decltype(a)::operator id<void(*)()>;
  private:
    using decltype(a)::operator id<void(^)()>;
  } extern c;

  struct D : decltype(a) {
    using decltype(a)::operator id<void(^)()>;
  private:
    using decltype(a)::operator id<void(*)()>; // expected-note {{here}}
  } extern d;

  bool r1 = c; // expected-warning{{address of lambda function pointer conversion operator will always evaluate to 'true'}}
  bool r2 = d; // expected-error {{private}} \
                  expected-warning{{address of lambda function pointer conversion operator will always evaluate to 'true'}}
}

namespace PR13117 {
  struct A {
    template<typename ... Args> static void f(Args...);

    template<typename ... Args> static void f1()
    {
      (void)^(Args args) { // expected-error{{expression contains unexpanded parameter pack 'Args'}}
      };
    }

    template<typename ... Args> static void f2()
    {
      f(
        ^(Args args)
        { }
        ...
      );
    }

    template<typename ... Args> static void f3()
    {
      (void)[](Args args) { // expected-error{{expression contains unexpanded parameter pack 'Args'}}
      };
    }

    template<typename ... Args> static void f4()
    {
      f([](Args args) { } ...);
    }
  };

  void g() {
    A::f1<int, int>();
    A::f2<int, int>();
  }
}
