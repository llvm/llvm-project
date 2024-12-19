// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedCallArgsChecker -verify %s

#include "mock-types.h"

CheckedObj* provide();
void consume_refcntbl(CheckedObj*);
void some_function();

namespace simple {
  void foo() {
    consume_refcntbl(provide());
    // expected-warning@-1{{Call argument is unchecked and unsafe}}
  }

  // Test that the checker works with [[clang::suppress]].
  void foo_suppressed() {
    [[clang::suppress]]
    consume_refcntbl(provide()); // no-warning
  }
}

namespace multi_arg {
  void consume_refcntbl(int, CheckedObj* foo, bool);
  void foo() {
    consume_refcntbl(42, provide(), true);
    // expected-warning@-1{{Call argument for parameter 'foo' is unchecked and unsafe}}
  }
}

namespace ref_counted {
  CheckedRef<CheckedObj> provide_ref_counted() { return CheckedRef<CheckedObj>{}; }
  void consume_ref_counted(CheckedRef<CheckedObj>) {}

  void foo() {
    consume_refcntbl(provide_ref_counted().ptr());
    // no warning
  }
}

namespace methods {
  struct Consumer {
    void consume_ptr(CheckedObj* ptr);
    void consume_ref(const CheckedObj& ref);
  };

  void foo() {
    Consumer c;

    c.consume_ptr(provide());
    // expected-warning@-1{{Call argument for parameter 'ptr' is unchecked and unsafe}}
    c.consume_ref(*provide());
    // expected-warning@-1{{Call argument for parameter 'ref' is unchecked and unsafe}}
  }

  void foo2() {
    struct Consumer {
      void consume(CheckedObj*) { some_function(); }
      void whatever() {
        consume(provide());
        // expected-warning@-1{{Call argument is unchecked and unsafe}}
      }
    };
  }

  void foo3() {
    struct Consumer {
      void consume(CheckedObj*) { some_function(); }
      void whatever() {
        this->consume(provide());
        // expected-warning@-1{{Call argument is unchecked and unsafe}}
      }
    };
  }
}

namespace casts {
  CheckedObj* downcast(CheckedObj*);

  void foo() {
    consume_refcntbl(provide());
    // expected-warning@-1{{Call argument is unchecked and unsafe}}

    consume_refcntbl(static_cast<CheckedObj*>(provide()));
    // expected-warning@-1{{Call argument is unchecked and unsafe}}

    consume_refcntbl(dynamic_cast<CheckedObj*>(provide()));
    // expected-warning@-1{{Call argument is unchecked and unsafe}}

    consume_refcntbl(const_cast<CheckedObj*>(provide()));
    // expected-warning@-1{{Call argument is unchecked and unsafe}}

    consume_refcntbl(reinterpret_cast<CheckedObj*>(provide()));
    // expected-warning@-1{{Call argument is unchecked and unsafe}}

    consume_refcntbl(downcast(provide()));
    // expected-warning@-1{{Call argument is unchecked and unsafe}}

    consume_refcntbl(
      static_cast<CheckedObj*>(
        downcast(
          static_cast<CheckedObj*>(
            provide()
          )
        )
      )
    );
    // expected-warning@-8{{Call argument is unchecked and unsafe}}
  }
}

namespace null_ptr {
  void foo_ref() {
    consume_refcntbl(nullptr);
    consume_refcntbl(0);
  }
}

namespace ref_counted_lookalike {
  struct Decoy {
    CheckedObj* get();
  };

  void foo() {
    Decoy D;

    consume_refcntbl(D.get());
    // expected-warning@-1{{Call argument is unchecked and unsafe}}
  }
}

namespace Ref_to_reference_conversion_operator {
  template<typename T> struct Ref {
    Ref() = default;
    Ref(T*) { }
    T* get() { return nullptr; }
    operator T& () { return t; }
    T t;
  };

  void consume_ref(CheckedObj&) {}

  void foo() {
    CheckedRef<CheckedObj> bar;
    consume_ref(bar);
  }
}

namespace param_formarding_function {
  void consume_ref_countable_ref(CheckedObj&);
  void consume_ref_countable_ptr(CheckedObj*);

  namespace ptr {
    void foo(CheckedObj* param) {
      consume_ref_countable_ptr(param);
    }
  }

  namespace ref {
    void foo(CheckedObj& param) {
      consume_ref_countable_ref(param);
    }
  }

  namespace ref_deref_operators {
    void foo_ref(CheckedObj& param) {
      consume_ref_countable_ptr(&param);
    }

    void foo_ptr(CheckedObj* param) {
      consume_ref_countable_ref(*param);
    }
  }

  namespace casts {

  CheckedObj* downcast(CheckedObj*) { return nullptr; }

  template<class T>
  T* bitwise_cast(T*) { return nullptr; }

    void foo(CheckedObj* param) {
      consume_ref_countable_ptr(downcast(param));
      consume_ref_countable_ptr(bitwise_cast(param));
     }
  }
}

namespace param_formarding_lambda {
  auto consume_ref_countable_ref = [](CheckedObj&) { some_function(); };
  auto consume_ref_countable_ptr = [](CheckedObj*) { some_function(); };

  namespace ptr {
    void foo(CheckedObj* param) {
      consume_ref_countable_ptr(param);
    }
  }

  namespace ref {
    void foo(CheckedObj& param) {
      consume_ref_countable_ref(param);
    }
  }

  namespace ref_deref_operators {
    void foo_ref(CheckedObj& param) {
      consume_ref_countable_ptr(&param);
    }

    void foo_ptr(CheckedObj* param) {
      consume_ref_countable_ref(*param);
    }
  }

  namespace casts {

  CheckedObj* downcast(CheckedObj*) { return nullptr; }

  template<class T>
  T* bitwise_cast(T*) { return nullptr; }

    void foo(CheckedObj* param) {
      consume_ref_countable_ptr(downcast(param));
      consume_ref_countable_ptr(bitwise_cast(param));
    }
  }
}

namespace param_forwarding_method {
  struct methodclass {
    void consume_ref_countable_ref(CheckedObj&) {};
    static void consume_ref_countable_ptr(CheckedObj*) {};
  };

  namespace ptr {
    void foo(CheckedObj* param) {
      methodclass::consume_ref_countable_ptr(param);
     }
  }

  namespace ref {
    void foo(CheckedObj& param) {
      methodclass mc;
      mc.consume_ref_countable_ref(param);
     }
  }

  namespace ref_deref_operators {
    void foo_ref(CheckedObj& param) {
      methodclass::consume_ref_countable_ptr(&param);
     }

    void foo_ptr(CheckedObj* param) {
      methodclass mc;
      mc.consume_ref_countable_ref(*param);
     }
  }

  namespace casts {

  CheckedObj* downcast(CheckedObj*) { return nullptr; }

  template<class T>
  T* bitwise_cast(T*) { return nullptr; }

    void foo(CheckedObj* param) {
      methodclass::consume_ref_countable_ptr(downcast(param));
       methodclass::consume_ref_countable_ptr(bitwise_cast(param));
     }
  }
}

namespace downcast {
  void consume_ref_countable(CheckedObj*) {}
  CheckedObj* downcast(CheckedObj*) { return nullptr; }

  void foo() {
    CheckedPtr<CheckedObj> bar;
    consume_ref_countable( downcast(bar.get()) );
  }
}

namespace string_impl {
  struct String {
    CheckedObj* impl() { return nullptr; }
  };

  struct AtomString {
    CheckedObj rc;
    CheckedObj& impl() { return rc; }
  };

  void consume_ptr(CheckedObj*) {}
  void consume_ref(CheckedObj&) {}

  namespace simple {
    void foo() {
      String s;
      AtomString as;
      consume_ptr(s.impl());
      consume_ref(as.impl());
    }
  }
}

namespace default_arg {
  CheckedObj* global;

  void function_with_default_arg(CheckedObj* param = global);
  // expected-warning@-1{{Call argument for parameter 'param' is unchecked and unsafe}}

  void foo() {
    function_with_default_arg();
  }
}

namespace cxx_member_func {
  CheckedRef<CheckedObj> provideProtected();
  void foo() {
    provide()->trivial();
    provide()->method();
    // expected-warning@-1{{Call argument for 'this' parameter is unchecked and unsafe}}
    provideProtected()->method();
    (provideProtected())->method();
  };
}

namespace cxx_member_operator_call {
  // The hidden this-pointer argument without a corresponding parameter caused couple bugs in parameter <-> argument attribution.
  struct Foo {
    Foo& operator+(CheckedObj* bad);
    friend Foo& operator-(Foo& lhs, CheckedObj* bad);
    void operator()(CheckedObj* bad);
  };

  CheckedObj* global;

  void foo() {
    Foo f;
    f + global;
    // expected-warning@-1{{Call argument for parameter 'bad' is unchecked and unsafe}}
    f - global;
    // expected-warning@-1{{Call argument for parameter 'bad' is unchecked and unsafe}}
    f(global);
    // expected-warning@-1{{Call argument for parameter 'bad' is unchecked and unsafe}}
  }
}

namespace call_with_ptr_on_ref {
  CheckedRef<CheckedObj> provideProtected();
  void bar(CheckedObj* bad);
  bool baz();
  void foo(bool v) {
    bar(v ? nullptr : provideProtected().ptr());
    bar(baz() ? provideProtected().ptr() : nullptr);
    bar(v ? provide() : provideProtected().ptr());
    // expected-warning@-1{{Call argument for parameter 'bad' is unchecked and unsafe}}
    bar(v ? provideProtected().ptr() : provide());
    // expected-warning@-1{{Call argument for parameter 'bad' is unchecked and unsafe}}
  }
}

namespace call_with_explicit_temporary_obj {
  void foo() {
    CheckedRef { *provide() }->method();
    CheckedPtr { provide() }->method();
  }
}

namespace call_with_checked_ptr {

  class Foo : public CheckedObj {
  public:
    CheckedPtr<CheckedObj> obj1() { return m_obj; }
    CheckedRef<CheckedObj> obj2() { return *m_obj; }
  private:
    CheckedObj* m_obj;
  };

  Foo* getFoo();

  void bar() {
    getFoo()->obj1()->method();
    getFoo()->obj2()->method();
  }

}
