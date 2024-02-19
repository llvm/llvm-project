// RUN: %check_clang_tidy %s modernize-use-override %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-override.IgnoreTemplateInstantiations: true}}"

struct Base {
  virtual void foo();
};

struct Base2 {
  virtual void foo2();
};

template<typename T>
struct Derived : T {
  // should not warn, comes from template instance
  virtual void foo();
  virtual void foo2();
};

void test() {
  Derived<Base> b;
  Derived<Base2> b2;
}

template<typename T>
struct BaseS {
  virtual void boo();
};

template<>
struct BaseS<int> {
  virtual void boo2();
};

struct BaseU {
  virtual void boo3();
};

template<typename T>
struct Derived2 : BaseS<T>, BaseU {
  // should not warn, comes from template instance
  virtual void boo();
  virtual void boo2();
  // should warn, comes from non-template BaseU
  virtual void boo3();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using 'override' or (rarely) 'final' instead of 'virtual' [modernize-use-override]
  // CHECK-FIXES: {{^  }}void boo3() override;
};
