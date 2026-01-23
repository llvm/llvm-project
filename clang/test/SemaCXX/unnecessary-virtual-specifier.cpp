// RUN: %clang_cc1 -fsyntax-only -verify -Wunnecessary-virtual-specifier -Wno-inconsistent-missing-override %s

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
  virtual int  virt2(int);
  virtual bool virt3(bool);
  int nonvirt();
};

struct Bar final : BarBase {
  ~Bar() override = delete;
          void virt() override {};
  virtual int  virt2(int) override;               // `virtual ... override;` is a common pattern, so don't warn
  virtual bool virt3(bool);                       // Already virtual in the base class; triggers
                                                  // -Winconsistent-missing-override or -Wsuggest-override instead
  virtual int  new_virt(bool);                    // expected-warning {{virtual method}}
  int nonvirt();
};
