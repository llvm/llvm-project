// RUN: %clang_cc1 -fsyntax-only -verify -Wunnecessary-virtual-specifier %s

struct Foo final {
  Foo() = default;
  virtual ~Foo() = default;                      // expected-warning {{virtual method}}
  virtual Foo& operator=(Foo& other) = default;  // expected-warning {{virtual method}}
  virtual Foo& operator=(Foo&& other) = default; // expected-warning {{virtual method}}
  void f();
  virtual void f(int);                           // expected-warning {{virtual method}}
  int g(int x) { return x; };
  virtual int g(bool);                           // expected-warning {{virtual method}}
  static int s();
};

struct BarBase {
  virtual ~BarBase() = delete;
  virtual void virt() {}
  virtual int virt(int);
  int nonvirt();
};

struct Bar final : BarBase {
  ~Bar() override = delete;
  void virt() override {};
  // `virtual ... override;` is a common pattern, so don't warn
  virtual int virt(int) override;
  int nonvirt();
};
