// RUN: %clang_cc1 -fsyntax-only -Wthread-safety -verify %s

class __attribute__((shared_capability("mutex"))) Mutex {
 public:
  void func1() __attribute__((assert_capability(this)));
  void func2() __attribute__((assert_capability(!this)));

  const Mutex& operator!() const { return *this; }
};

class NotACapability {
 public:
  void func1() __attribute__((assert_capability(this)));  // expected-warning {{'assert_capability' attribute requires arguments whose type is annotated with 'capability' attribute; type here is 'NotACapability *'}}
  void func2() __attribute__((assert_capability(!this)));  // expected-warning {{'assert_capability' attribute requires arguments whose type is annotated with 'capability' attribute; type here is 'bool'}}

  const NotACapability& operator!() const { return *this; }
};
