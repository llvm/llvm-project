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
