// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

RefCountable* provide() { return nullptr; }
void consume_refcntbl(RefCountable*) {}

namespace simple {
  void foo() {
    consume_refcntbl(provide());
    // expected-warning@-1{{Call argument is uncounted and unsafe}}
  }
}

namespace multi_arg {
  void consume_refcntbl(int, RefCountable* foo, bool) {}
  void foo() {
    consume_refcntbl(42, provide(), true);
    // expected-warning@-1{{Call argument for parameter 'foo' is uncounted and unsafe}}
  }
}

namespace ref_counted {
  Ref<RefCountable> provide_ref_counted() { return Ref<RefCountable>{}; }
  void consume_ref_counted(Ref<RefCountable>) {}

  void foo() {
    consume_refcntbl(provide_ref_counted().get());
    // no warning
  }
}

namespace methods {
  struct Consumer {
    void consume_ptr(RefCountable* ptr) {}
    void consume_ref(const RefCountable& ref) {}
  };

  void foo() {
    Consumer c;

    c.consume_ptr(provide());
    // expected-warning@-1{{Call argument for parameter 'ptr' is uncounted and unsafe}}
    c.consume_ref(*provide());
    // expected-warning@-1{{Call argument for parameter 'ref' is uncounted and unsafe}}
  }

  void foo2() {
    struct Consumer {
      void consume(RefCountable*) { }
      void whatever() {
        consume(provide());
        // expected-warning@-1{{Call argument is uncounted and unsafe}}
      }
    };
  }

  void foo3() {
    struct Consumer {
      void consume(RefCountable*) { }
      void whatever() {
        this->consume(provide());
        // expected-warning@-1{{Call argument is uncounted and unsafe}}
      }
    };
  }
}

namespace casts {
  RefCountable* downcast(RefCountable*) { return nullptr; }

  void foo() {
    consume_refcntbl(provide());
    // expected-warning@-1{{Call argument is uncounted and unsafe}}

    consume_refcntbl(static_cast<RefCountable*>(provide()));
    // expected-warning@-1{{Call argument is uncounted and unsafe}}

    consume_refcntbl(dynamic_cast<RefCountable*>(provide()));
    // expected-warning@-1{{Call argument is uncounted and unsafe}}

    consume_refcntbl(const_cast<RefCountable*>(provide()));
    // expected-warning@-1{{Call argument is uncounted and unsafe}}

    consume_refcntbl(reinterpret_cast<RefCountable*>(provide()));
    // expected-warning@-1{{Call argument is uncounted and unsafe}}

    consume_refcntbl(downcast(provide()));
    // expected-warning@-1{{Call argument is uncounted and unsafe}}

    consume_refcntbl(
      static_cast<RefCountable*>(
        downcast(
          static_cast<RefCountable*>(
            provide()
          )
        )
      )
    );
    // expected-warning@-8{{Call argument is uncounted and unsafe}}
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
    RefCountable* get() { return nullptr; }
  };

  void foo() {
    Decoy D;

    consume_refcntbl(D.get());
    // expected-warning@-1{{Call argument is uncounted and unsafe}}
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

  void consume_ref(RefCountable&) {}

  void foo() {
    Ref<RefCountable> bar;
    consume_ref(bar);
  }
}

namespace param_formarding_function {
  void consume_ref_countable_ref(RefCountable&) {}
  void consume_ref_countable_ptr(RefCountable*) {}

  namespace ptr {
    void foo(RefCountable* param) {
      consume_ref_countable_ptr(param);
    }
  }

  namespace ref {
    void foo(RefCountable& param) {
      consume_ref_countable_ref(param);
    }
  }

  namespace ref_deref_operators {
    void foo_ref(RefCountable& param) {
      consume_ref_countable_ptr(&param);
    }

    void foo_ptr(RefCountable* param) {
      consume_ref_countable_ref(*param);
    }
  }

  namespace casts {

  RefCountable* downcast(RefCountable*) { return nullptr; }

  template<class T>
  T* bitwise_cast(T*) { return nullptr; }

    void foo(RefCountable* param) {
      consume_ref_countable_ptr(downcast(param));
      consume_ref_countable_ptr(bitwise_cast(param));
     }
  }
}

namespace param_formarding_lambda {
  auto consume_ref_countable_ref = [](RefCountable&) {};
  auto consume_ref_countable_ptr = [](RefCountable*) {};

  namespace ptr {
    void foo(RefCountable* param) {
      consume_ref_countable_ptr(param);
    }
  }

  namespace ref {
    void foo(RefCountable& param) {
      consume_ref_countable_ref(param);
    }
  }

  namespace ref_deref_operators {
    void foo_ref(RefCountable& param) {
      consume_ref_countable_ptr(&param);
    }

    void foo_ptr(RefCountable* param) {
      consume_ref_countable_ref(*param);
    }
  }

  namespace casts {

  RefCountable* downcast(RefCountable*) { return nullptr; }

  template<class T>
  T* bitwise_cast(T*) { return nullptr; }

    void foo(RefCountable* param) {
      consume_ref_countable_ptr(downcast(param));
      consume_ref_countable_ptr(bitwise_cast(param));
    }
  }
}

namespace param_forwarding_method {
  struct methodclass {
    void consume_ref_countable_ref(RefCountable&) {};
    static void consume_ref_countable_ptr(RefCountable*) {};
  };

  namespace ptr {
    void foo(RefCountable* param) {
      methodclass::consume_ref_countable_ptr(param);
     }
  }

  namespace ref {
    void foo(RefCountable& param) {
      methodclass mc;
      mc.consume_ref_countable_ref(param);
     }
  }

  namespace ref_deref_operators {
    void foo_ref(RefCountable& param) {
      methodclass::consume_ref_countable_ptr(&param);
     }

    void foo_ptr(RefCountable* param) {
      methodclass mc;
      mc.consume_ref_countable_ref(*param);
     }
  }

  namespace casts {

  RefCountable* downcast(RefCountable*) { return nullptr; }

  template<class T>
  T* bitwise_cast(T*) { return nullptr; }

    void foo(RefCountable* param) {
      methodclass::consume_ref_countable_ptr(downcast(param));
       methodclass::consume_ref_countable_ptr(bitwise_cast(param));
     }
  }
}

namespace downcast {
  void consume_ref_countable(RefCountable*) {}
  RefCountable* downcast(RefCountable*) { return nullptr; }

  void foo() {
    RefPtr<RefCountable> bar;
    consume_ref_countable( downcast(bar.get()) );
  }
}

namespace string_impl {
  struct String {
    RefCountable* impl() { return nullptr; }
  };

  struct AtomString {
    RefCountable rc;
    RefCountable& impl() { return rc; }
  };

  void consume_ptr(RefCountable*) {}
  void consume_ref(RefCountable&) {}

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
  RefCountable* global;

  void function_with_default_arg(RefCountable* param = global) {}
  // expected-warning@-1{{Call argument for parameter 'param' is uncounted and unsafe}}

  void foo() {
    function_with_default_arg();
  }
}

namespace cxx_member_operator_call {
  // The hidden this-pointer argument without a corresponding parameter caused couple bugs in parameter <-> argument attribution.
  struct Foo {
    Foo& operator+(RefCountable* bad) { return *this; }
    friend Foo& operator-(Foo& lhs, RefCountable* bad) { return lhs; }
    void operator()(RefCountable* bad) { }
  };

  RefCountable* global;

  void foo() {
    Foo f;
    f + global;
    // expected-warning@-1{{Call argument for parameter 'bad' is uncounted and unsafe}}
    f - global;
    // expected-warning@-1{{Call argument for parameter 'bad' is uncounted and unsafe}}
    f(global);
    // expected-warning@-1{{Call argument for parameter 'bad' is uncounted and unsafe}}
  }
}
