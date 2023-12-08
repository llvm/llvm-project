// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14

namespace access_control {
class Private {
  void check(int *) __attribute__((enable_if(false, "")));
  void check(double *) __attribute__((enable_if(true, "")));

  static void checkStatic(int *) __attribute__((enable_if(false, "")));
  static void checkStatic(double *) __attribute__((enable_if(true, "")));
};

auto Priv = reinterpret_cast<void (Private::*)(char *)>(&Private::check); // expected-error{{'check' is a private member of 'access_control::Private'}} expected-note@6{{implicitly declared private here}}

auto PrivStatic = reinterpret_cast<void (*)(char *)>(&Private::checkStatic); // expected-error{{'checkStatic' is a private member of 'access_control::Private'}} expected-note@9{{implicitly declared private here}}

class Protected {
protected:
  void check(int *) __attribute__((enable_if(false, "")));
  void check(double *) __attribute__((enable_if(true, "")));

  static void checkStatic(int *) __attribute__((enable_if(false, "")));
  static void checkStatic(double *) __attribute__((enable_if(true, "")));
};

auto Prot = reinterpret_cast<void (Protected::*)(char *)>(&Protected::check); // expected-error{{'check' is a protected member of 'access_control::Protected'}} expected-note@19{{declared protected here}}

auto ProtStatic = reinterpret_cast<void (*)(char *)>(&Protected::checkStatic); // expected-error{{'checkStatic' is a protected member of 'access_control::Protected'}} expected-note@22{{declared protected here}}
}

namespace unavailable {
// Ensure that we check that the function can be called
void foo() __attribute__((unavailable("don't call this")));
void foo(int) __attribute__((enable_if(false, "")));

void *Ptr = reinterpret_cast<void*>(foo); // expected-error{{'foo' is unavailable: don't call this}} expected-note@-3{{explicitly marked unavailable here}}
}

namespace template_deduction {
void foo() __attribute__((enable_if(false, "")));

void bar() __attribute__((enable_if(true, "")));
void bar() __attribute__((enable_if(false, "")));

void baz(int a) __attribute__((enable_if(true, "")));
void baz(int a) __attribute__((enable_if(a, "")));
void baz(int a) __attribute__((enable_if(false, "")));

void qux(int a) __attribute__((enable_if(1, "")));
void qux(int a) __attribute__((enable_if(true, "")));
void qux(int a) __attribute__((enable_if(a, "")));
void qux(int a) __attribute__((enable_if(false, "")));

template <typename Fn, typename... Args> void call(Fn F, Args... As) {
  F(As...);
}

void test() {
  call(foo); // expected-error{{cannot take address of function 'foo'}}
  call(bar);
  call(baz, 0);
  call(qux, 0); // expected-error{{no matching function for call to 'call'}} expected-note@53{{candidate template ignored: couldn't infer template argument 'Fn'}}

  auto Ptr1 = foo; // expected-error{{cannot take address of function 'foo'}}
  auto Ptr2 = bar;
  auto Ptr3 = baz;
  auto Ptr4 = qux; // expected-error{{variable 'Ptr4' with type 'auto' has incompatible initializer of type '<overloaded function type>'}}
}

template <typename Fn, typename T, typename... Args>
void callMem(Fn F, T t, Args... As) {
  (t.*F)(As...);
}

class Foo {
  void bar() __attribute__((enable_if(true, "")));
  void bar() __attribute__((enable_if(false, "")));

  static void staticBar() __attribute__((enable_if(true, "")));
  static void staticBar() __attribute__((enable_if(false, "")));
};

void testAccess() {
  callMem(&Foo::bar, Foo()); // expected-error{{'bar' is a private member of 'template_deduction::Foo'}} expected-note@-8{{implicitly declared private here}}
  call(&Foo::staticBar); // expected-error{{'staticBar' is a private member of 'template_deduction::Foo'}} expected-note@-6{{implicitly declared private here}}
}
}

namespace template_template_deduction {
void foo() __attribute__((enable_if(false, "")));
template <typename T>
T foo() __attribute__((enable_if(true, "")));

template <typename Fn, typename... Args> auto call(Fn F, Args... As) {
  return F(As...);
}

auto Ok = call(&foo<int>);
auto Fail = call(&foo); // expected-error{{no matching function for call to 'call'}} expected-note@-5{{candidate template ignored: couldn't infer template argument 'Fn'}}

auto PtrOk = &foo<int>;
auto PtrFail = &foo; // expected-error{{variable 'PtrFail' with type 'auto' has incompatible initializer of type '<overloaded function type>'}}
}

namespace pointer_equality {
  using FnTy = void (*)();

  void bothEnableIf() __attribute__((enable_if(false, "")));
  void bothEnableIf() __attribute__((enable_if(true, "")));

  void oneEnableIf() __attribute__((enable_if(false, "")));
  void oneEnableIf();

  void test() {
    FnTy Fn;
    (void)(Fn == bothEnableIf);
    (void)(Fn == &bothEnableIf);
    (void)(Fn == oneEnableIf);
    (void)(Fn == &oneEnableIf);
  }

  void unavailableEnableIf() __attribute__((enable_if(false, "")));
  void unavailableEnableIf() __attribute__((unavailable("noooo"))); // expected-note 2{{marked unavailable here}}

  void testUnavailable() {
    FnTy Fn;
    (void)(Fn == unavailableEnableIf); // expected-error{{is unavailable}}
    (void)(Fn == &unavailableEnableIf); // expected-error{{is unavailable}}
  }

  class Foo {
    static void staticAccessEnableIf(); // expected-note 2{{declared private here}}
    void accessEnableIf(); // expected-note{{declared private here}}

  public:
    static void staticAccessEnableIf() __attribute__((enable_if(false, "")));
    void accessEnableIf() __attribute__((enable_if(false, "")));
  };

  void testAccess() {
    FnTy Fn;
    (void)(Fn == Foo::staticAccessEnableIf); // expected-error{{is a private member}}
    (void)(Fn == &Foo::staticAccessEnableIf); // expected-error{{is a private member}}

    void (Foo::*MemFn)();
    (void)(MemFn == &Foo::accessEnableIf); // expected-error{{is a private member}}
  }
}
