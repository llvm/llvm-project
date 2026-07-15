// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

RefCountable* provide();
void consume_refcntbl(RefCountable*);
void some_function();

namespace simple {
  void foo() {
    consume_refcntbl(provide());
    // expected-warning@-1{{Function argument 'provide()' (to 'consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  }

  // Test that the checker works with [[clang::suppress]].
  void foo_suppressed() {
    [[clang::suppress]]
    consume_refcntbl(provide()); // no-warning
  }
}

namespace multi_arg {
  void consume_refcntbl(int, RefCountable* foo, bool);
  void foo() {
    consume_refcntbl(42, provide(), true);
    // expected-warning@-1{{Function argument 'provide()' (parameter 'foo' to 'multi_arg::consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  }
}

namespace ref_counted {
  Ref<RefCountable> provide_ref_counted() { return Ref<RefCountable>{}; }
  void consume_ref_counted(Ref<RefCountable>) {}

  void foo() {
    consume_refcntbl(provide_ref_counted().ptr());
    // no warning
  }
}

namespace methods {
  struct Consumer {
    void consume_ptr(RefCountable* ptr);
    void consume_ref(const RefCountable& ref);
  };

  void foo() {
    Consumer c;

    c.consume_ptr(provide());
    // expected-warning@-1{{Function argument 'provide()' (parameter 'ptr' to 'methods::Consumer::consume_ptr') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    c.consume_ref(*provide());
    // expected-warning@-1{{Function argument '*provide()' (parameter 'ref' to 'methods::Consumer::consume_ref') is a raw reference to RefPtr-capable type}}
  }

  void foo2() {
    struct Consumer {
      void consume(RefCountable*) { some_function(); }
      void whatever() {
        consume(provide());
        // expected-warning@-1{{Function argument 'provide()' (to 'methods::foo2()::Consumer::consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
      }
    };
  }

  void foo3() {
    struct Consumer {
      void consume(RefCountable*) { some_function(); }
      void whatever() {
        this->consume(provide());
        // expected-warning@-1{{Function argument 'provide()' (to 'methods::foo3()::Consumer::consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
      }
    };
  }
}

namespace casts {
  RefCountable* downcast(RefCountable*);

  void foo() {
    consume_refcntbl(provide());
    // expected-warning@-1{{Function argument 'provide()' (to 'consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}

    consume_refcntbl(static_cast<RefCountable*>(provide()));
    // expected-warning@-1{{Function argument 'static_cast<RefCountable *>(provide())' (to 'consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}

    consume_refcntbl(dynamic_cast<RefCountable*>(provide()));
    // expected-warning@-1{{Function argument 'dynamic_cast<RefCountable *>(provide())' (to 'consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}

    consume_refcntbl(const_cast<RefCountable*>(provide()));
    // expected-warning@-1{{Function argument 'const_cast<RefCountable *>(provide())' (to 'consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}

    consume_refcntbl(reinterpret_cast<RefCountable*>(provide()));
    // expected-warning@-1{{Function argument 'reinterpret_cast<RefCountable *>(provide())' (to 'consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}

    consume_refcntbl(downcast(provide()));
    // expected-warning@-1{{Function argument 'downcast(provide())' (to 'consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}

    consume_refcntbl(
      static_cast<RefCountable*>(
        downcast(
          static_cast<RefCountable*>(
            provide()
          )
        )
      )
    );
    // expected-warning@-8{{Function argument 'static_cast<RefCountable *>(downcast(static_cast<R...' (to 'consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}
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
    RefCountable* get();
  };

  void foo() {
    Decoy D;

    consume_refcntbl(D.get());
    // expected-warning@-1{{Function argument 'D.get()' (to 'consume_refcntbl') is a raw pointer to RefPtr-capable type 'RefCountable'}}
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
  void consume_ref_countable_ref(RefCountable&);
  void consume_ref_countable_ptr(RefCountable*);

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

  RefCountable* downcast(RefCountable*);
  template<class T> T* bitwise_cast(T*);
  template<class T> T* bit_cast(T*);

  void foo(RefCountable* param) {
    consume_ref_countable_ptr(downcast(param));
    consume_ref_countable_ptr(bitwise_cast(param));
    consume_ref_countable_ptr(bit_cast(param));
   }
  }
}

namespace param_formarding_lambda {
  auto consume_ref_countable_ref = [](RefCountable&) { some_function(); };
  auto consume_ref_countable_ptr = [](RefCountable*) { some_function(); };

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

  void function_with_default_arg(RefCountable* param = global);
  // expected-warning@-1{{Function argument 'global' (parameter 'param' to 'default_arg::function_with_default_arg') is a raw pointer to RefPtr-capable type 'RefCountable'}}

  void foo() {
    function_with_default_arg();
  }
}

namespace cxx_member_func {
  Ref<RefCountable> provideProtected();
  void foo() {
    provide()->trivial();
    provide()->method();
    // expected-warning@-1{{Function argument 'provide()' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    provideProtected()->method();
    (provideProtected())->method();
  };
}

namespace cxx_member_operator_call {
  // The hidden this-pointer argument without a corresponding parameter caused couple bugs in parameter <-> argument attribution.
  struct Foo {
    Foo& operator+(RefCountable* bad);
    friend Foo& operator-(Foo& lhs, RefCountable* bad);
    void operator()(RefCountable* bad);
    int operator[](unsigned i);
  };

  RefCountable* global;

  struct Container : public RefCountable {
    int m[4];
    int& operator[](unsigned i) {
      some_function();
      return m[i];
    }
  };
  Container container();

  void foo12() {
    Foo f;
    f + global;
    // expected-warning@-1{{Function argument 'global' (parameter 'bad' to 'cxx_member_operator_call::Foo::operator+') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    f - global;
    // expected-warning@-1{{Function argument 'global' (parameter 'bad' to 'cxx_member_operator_call::operator-') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    f(global);
    // expected-warning@-1{{Function argument 'global' (parameter 'bad' to 'cxx_member_operator_call::Foo::operator()') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    container()[0] = 3;
    // expected-warning@-1{{Function argument 'container()' (parameter 'this' to 'cxx_member_operator_call::Container::operator[]') is a raw pointer to RefPtr-capable type 'cxx_member_operator_call::Container'}}
  }
}

namespace call_function_ptr {
  class RefCountableWithWeakPtr : public RefCountable, public CanMakeWeakPtr<RefCountableWithWeakPtr> {
  public:
    void method();
  };

  RefCountableWithWeakPtr* provide();

  void foo(void (*consume)(void*, RefCountableWithWeakPtr*), void (*consumeVar)(RefCountableWithWeakPtr*, ...), void (RefCountableWithWeakPtr::*method)()) {
    consume(nullptr, provide());
    // expected-warning@-1{{Function argument 'provide()' is a raw pointer to RefPtr-capable type 'call_function_ptr::RefCountableWithWeakPtr}}
    consumeVar(nullptr, provide());
    // expected-warning@-1{{Function argument 'provide()' is a raw pointer to RefPtr-capable type 'call_function_ptr::RefCountableWithWeakPtr}}
    (provide()->*method)();
    // expected-warning@-1{{Function argument 'provide()' (parameter 'this') is a raw pointer to RefPtr-capable type 'call_function_ptr::RefCountableWithWeakPtr'}}
  }

  template <typename T, typename U, typename... Arg>
  void bar(T* obj, int (U::* function)(void*, Arg... args), Arg... args) {
    (obj->*function)(args...);
  }
}

namespace call_with_ptr_on_ref {
  Ref<RefCountable> provideProtected();
  void bar(RefCountable* bad);
  bool baz();
  void foo(bool v) {
    bar(v ? nullptr : provideProtected().ptr());
    bar(baz() ? provideProtected().ptr() : nullptr);
    bar(v ? provide() : provideProtected().ptr());
    // expected-warning@-1{{Function argument 'v ? provide() : provideProtected().ptr()' (parameter 'bad' to 'call_with_ptr_on_ref::bar') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    bar(v ? provideProtected().ptr() : provide());
    // expected-warning@-1{{Function argument 'v ? provideProtected().ptr() : provide()' (parameter 'bad' to 'call_with_ptr_on_ref::bar') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  }
}

namespace call_with_explicit_construct_from_auto {

  struct Impl {
    void ref() const;
    void deref() const;

    static Ref<Impl> create();
  };

  template <typename T>
  struct ArgObj {
    T* t;
  };

  struct Object {
    Object();
    Object(Ref<Impl>&&);

    Impl* impl() const { return m_impl.get(); }

    static Object create(ArgObj<char>&) { return Impl::create(); }
    static void bar(Impl&);

  private:
    RefPtr<Impl> m_impl;
  };

  template<typename CharacterType> void foo()
  {
      auto result = Object::create(ArgObj<CharacterType> { });
      Object::bar(Ref { *result.impl() });
  }

}

namespace call_with_explicit_temporary_obj {
  void foo() {
    Ref { *provide() }->method();
    RefPtr { provide() }->method();
  }
  template <typename T>
  void bar() {
    Ref(*provide())->method();
    RefPtr(provide())->method();
  }
  void baz() {
    bar<int>();
  }

  class Foo {
    Ref<RefCountable> ensure();
    void foo() {
      Ref { ensure() }->method();
    }
  };

  void baz(Ref<RefCountable>&& arg) {
    Ref { arg }->method();
  }
}

namespace call_with_explicit_construct {
  class Obj {
  public:
    Obj(RefCountable* obj = provide(), RefCountable* otherObj = nullptr) {
      // expected-warning@-1{{Function argument 'provide()' (parameter 'obj' to 'call_with_explicit_construct::Obj::Obj') is a raw pointer to RefPtr-capable type 'RefCountable'}}
      consume_refcntbl(obj);
      if (otherObj)
        otherObj->method();
    }
    Obj(RefCountable& obj) {
      obj.method();
    }
  };

  void foo(RefCountable* arg) {
    Obj obj1;
    Obj obj2(provide());
    // expected-warning@-1{{Function argument 'provide()' (parameter 'obj' to 'call_with_explicit_construct::Obj::Obj') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    Obj obj3(*provide());
    // expected-warning@-1{{Function argument '*provide()' (parameter 'obj' to 'call_with_explicit_construct::Obj::Obj') is a raw reference to RefPtr-capable type 'RefCountable'}}
    Obj obj4(Ref<RefCountable> { *provide() }.get());
    Obj obj5(arg);
    Obj obj6(arg, provide());
    // expected-warning@-1{{Function argument 'provide()' (parameter 'otherObj' to 'call_with_explicit_construct::Obj::Obj') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  }
}

namespace call_with_adopt_ref {
  class Obj {
  public:
    void ref() const;
    void deref() const;
    void method();
  };

  // This is needed due to rdar://141692212.
  struct dummy {
    RefPtr<Obj> any;
  };

  void foo() {
    adoptRef(new Obj)->method();
  }
}

namespace call_on_member {

  class SomeObj {
  public:
    static Ref<SomeObj> create() { return adoptRef(*new SomeObj); }

    void ref() const;
    void deref() const;

    void doWork() {
      m_obj->method();
      // expected-warning@-1{{Function argument 'this->m_obj' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
      m_obj.get()->method();
      // expected-warning@-1{{Function argument 'this->m_obj.get()' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
      m_constObj->method();
    }

    void localWork() {
      RefPtr obj = provide();
      obj->method();
      obj.get()->method();
    }

    void argWork(RefPtr<RefCountable> arg) {
      arg->method();
      arg.get()->method();
    }

    void temporaryWork() {
      RefPtr { provide() }->method();
      RefPtr { provide() }.get()->method();
    }

    void work();

    RefCountable& constObj() const { return *m_constObj; }

  private:
    RefPtr<RefCountable> m_obj;
    const RefPtr<RefCountable> m_constObj;
  };

  SomeObj* provide();

  void foo() {
    provide()->constObj().method();
    // expected-warning@-1{{Function argument 'provide()->constObj()' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    Ref { provide()->constObj() }->method();
    RefPtr { provide() }->constObj().method();
  }

}

namespace call_with_weak_ptr {

  class RefCountableWithWeakPtr : public RefCountable, public CanMakeWeakPtr<RefCountableWithWeakPtr> {
  };

  RefCountableWithWeakPtr* provide();
  void consume(RefCountableWithWeakPtr*);

  void foo() {
    WeakPtr weakPtr = provide();
    consume(weakPtr);
    // expected-warning@-1{{Function argument 'weakPtr' (to 'call_with_weak_ptr::consume') is a raw pointer to RefPtr-capable type 'call_with_weak_ptr::RefCountableWithWeakPtr'}}
    weakPtr->method();
    // expected-warning@-1{{Function argument 'weakPtr' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  }

  struct Provider {
    RefCountableWithWeakPtr* provide();
  };
  int intValue();

  struct Container {
    Container(Provider& provider)
      : m_weakPtr(provider.provide())
      , m_value(intValue())
    { }

  private:
    WeakPtr<RefCountableWithWeakPtr> m_weakPtr;
    int m_value;
  };

}
