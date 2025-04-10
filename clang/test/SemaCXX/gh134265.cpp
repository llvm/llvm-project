// RUN: %clang_cc1 %s -verify=expected -fsyntax-only -triple=x86_64-unknown-linux-gnu
// RUN: %clang_cc1 %s -verify=expected -fsyntax-only -triple=x86_64-unknown-linux-gnu -std=c++20
// RUN: %clang_cc1 %s -verify=expected,ms -fms-extensions -fms-compatibility -triple=x86_64-pc-windows-msvc -DMS

struct Foo {
  virtual ~Foo() {} // expected-error {{attempt to use a deleted function}}
  static void operator delete(void* ptr) = delete; // expected-note {{explicitly marked deleted here}}
};


struct Bar {
  virtual ~Bar() {}
  static void operator delete[](void* ptr) = delete;
};

struct Baz {
  virtual ~Baz() {}
  static void operator delete[](void* ptr) = delete; // expected-note {{explicitly marked deleted here}}\
                                                        ms-note{{explicitly marked deleted here}}}
};

struct BarBaz {
  ~BarBaz() {}
  static void operator delete[](void* ptr) = delete;
};

void foobar() {
  Baz *B = new Baz[10](); // ms-error {{attempt to use a deleted function}}
  delete [] B; // expected-error {{attempt to use a deleted function}}
  BarBaz *BB = new BarBaz[10]();
}

struct BaseDelete1 {
  void operator delete[](void *); //ms-note 3{{member found by ambiguous name lookup}}
};
struct BaseDelete2 {
  void operator delete[](void *); //ms-note 3{{member found by ambiguous name lookup}}
};
struct BaseDestructor {
  BaseDestructor() {}
  virtual ~BaseDestructor() = default;
};
struct Final : BaseDelete1, BaseDelete2, BaseDestructor {
  Final() {}
};
struct FinalExplicit : BaseDelete1, BaseDelete2, BaseDestructor {
  FinalExplicit() {}
  inline ~FinalExplicit() {}
};

#ifdef MS
struct Final1 : BaseDelete1, BaseDelete2, BaseDestructor {
  __declspec(dllexport) ~Final1() {} // ms-error {{member 'operator delete[]' found in multiple base classes of different types}}
};
#endif // MS

void foo() {
    Final* a = new Final[10](); // ms-error {{member 'operator delete[]' found in multiple base classes of different types}}
    FinalExplicit* b = new FinalExplicit[10](); // ms-error {{member 'operator delete[]' found in multiple base classes of different types}}
}
