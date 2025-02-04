// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.UncountedLambdaCapturesChecker -verify %s

#include "mock-types.h"

namespace WTF {

namespace Detail {

template<typename Out, typename... In>
class CallableWrapperBase {
public:
    virtual ~CallableWrapperBase() { }
    virtual Out call(In...) = 0;
};

template<typename, typename, typename...> class CallableWrapper;

template<typename CallableType, typename Out, typename... In>
class CallableWrapper : public CallableWrapperBase<Out, In...> {
public:
    explicit CallableWrapper(CallableType& callable)
        : m_callable(callable) { }
    Out call(In... in) final { return m_callable(in...); }

private:
    CallableType m_callable;
};

} // namespace Detail

template<typename> class Function;

template<typename Out, typename... In> Function<Out(In...)> adopt(Detail::CallableWrapperBase<Out, In...>*);

template <typename Out, typename... In>
class Function<Out(In...)> {
public:
    using Impl = Detail::CallableWrapperBase<Out, In...>;

    Function() = default;

    template<typename FunctionType>
    Function(FunctionType f)
        : m_callableWrapper(new Detail::CallableWrapper<FunctionType, Out, In...>(f)) { }

    Out operator()(In... in) const { return m_callableWrapper->call(in...); }
    explicit operator bool() const { return !!m_callableWrapper; }

private:
    enum AdoptTag { Adopt };
    Function(Impl* impl, AdoptTag)
        : m_callableWrapper(impl)
    {
    }

    friend Function adopt<Out, In...>(Impl*);

    std::unique_ptr<Impl> m_callableWrapper;
};

template<typename Out, typename... In> Function<Out(In...)> adopt(Detail::CallableWrapperBase<Out, In...>* impl)
{
    return Function<Out(In...)>(impl, Function<Out(In...)>::Adopt);
}

} // namespace WTF

struct A {
  static void b();
};

RefCountable* make_obj();

void someFunction();
template <typename Callback> void call(Callback callback) {
  someFunction();
  callback();
}

void raw_ptr() {
  RefCountable* ref_countable = make_obj();
  auto foo1 = [ref_countable](){
    // expected-warning@-1{{Captured raw-pointer 'ref_countable' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
    ref_countable->method();
  };
  auto foo2 = [&ref_countable](){
    // expected-warning@-1{{Captured raw-pointer 'ref_countable' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
    ref_countable->method();
  };
  auto foo3 = [&](){
    ref_countable->method();
    // expected-warning@-1{{Implicitly captured raw-pointer 'ref_countable' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
    ref_countable = nullptr;
  };

  auto foo4 = [=](){
    ref_countable->method();
    // expected-warning@-1{{Implicitly captured raw-pointer 'ref_countable' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
  };

  call(foo1);
  call(foo2);
  call(foo3);
  call(foo4);

  // Confirm that the checker respects [[clang::suppress]].
  RefCountable* suppressed_ref_countable = nullptr;
  [[clang::suppress]] auto foo5 = [suppressed_ref_countable](){};
  // no warning.
  call(foo5);
}

void references() {
  RefCountable automatic;
  RefCountable& ref_countable_ref = automatic;
  auto foo1 = [ref_countable_ref](){ ref_countable_ref.constMethod(); };
  // expected-warning@-1{{Captured reference 'ref_countable_ref' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
  auto foo2 = [&ref_countable_ref](){ ref_countable_ref.method(); };
  // expected-warning@-1{{Captured reference 'ref_countable_ref' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
  auto foo3 = [&](){ ref_countable_ref.method(); };
  // expected-warning@-1{{Implicitly captured reference 'ref_countable_ref' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
  auto foo4 = [=](){ ref_countable_ref.constMethod(); };
  // expected-warning@-1{{Implicitly captured reference 'ref_countable_ref' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}

  call(foo1);
  call(foo2);
  call(foo3);
  call(foo4);
}

void quiet() {
// This code is not expected to trigger any warnings.
  {
    RefCountable automatic;
    RefCountable &ref_countable_ref = automatic;
  }

  auto foo3 = [&]() {};
  auto foo4 = [=]() {};

  call(foo3);
  call(foo4);

  RefCountable *ref_countable = nullptr;
}

template <typename Callback>
void map(RefCountable* start, [[clang::noescape]] Callback&& callback)
{
  while (start) {
    callback(*start);
    start = start->next();
  }
}

template <typename Callback1, typename Callback2>
void doubleMap(RefCountable* start, [[clang::noescape]] Callback1&& callback1, Callback2&& callback2)
{
  while (start) {
    callback1(*start);
    callback2(*start);
    start = start->next();
  }
}

void noescape_lambda() {
  RefCountable* someObj = make_obj();
  RefCountable* otherObj = make_obj();
  map(make_obj(), [&](RefCountable& obj) {
    otherObj->method();
  });
  doubleMap(make_obj(), [&](RefCountable& obj) {
    otherObj->method();
  }, [&](RefCountable& obj) {
    otherObj->method();
    // expected-warning@-1{{Implicitly captured raw-pointer 'otherObj' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
  });
  ([&] {
    someObj->method();
  })();
}

void lambda_capture_param(RefCountable* obj) {
  auto someLambda = [&]() {
    obj->method();
  };
  someLambda();
  someLambda();
}

struct RefCountableWithLambdaCapturingThis {
  void ref() const;
  void deref() const;
  void nonTrivial();

  void method_captures_this_safe() {
    auto lambda = [&]() {
      nonTrivial();
    };
    lambda();
  }

  void method_captures_this_unsafe() {
    auto lambda = [&]() {
      nonTrivial();
      // expected-warning@-1{{Implicitly captured raw-pointer 'this' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
    };
    call(lambda);
  }

  void method_captures_this_unsafe_capture_local_var_explicitly() {
    RefCountable* x = make_obj();
    call([this, protectedThis = RefPtr { this }, x]() {
      // expected-warning@-1{{Captured raw-pointer 'x' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
      nonTrivial();
      x->method();
    });
  }

  void method_captures_this_with_other_protected_var() {
    RefCountable* x = make_obj();
    call([this, protectedX = RefPtr { x }]() {
      // expected-warning@-1{{Captured raw-pointer 'this' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
      nonTrivial();
      protectedX->method();
    });
  }

  void method_captures_this_unsafe_capture_local_var_explicitly_with_deref() {
    RefCountable* x = make_obj();
    call([this, protectedThis = Ref { *this }, x]() {
      // expected-warning@-1{{Captured raw-pointer 'x' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
      nonTrivial();
      x->method();
    });
  }

  void method_captures_this_unsafe_local_var_via_vardecl() {
    RefCountable* x = make_obj();
    auto lambda = [this, protectedThis = Ref { *this }, x]() {
      // expected-warning@-1{{Captured raw-pointer 'x' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
      nonTrivial();
      x->method();
    };
    call(lambda);
  }

  void method_captures_this_with_guardian() {
    auto lambda = [this, protectedThis = Ref { *this }]() {
      nonTrivial();
    };
    call(lambda);
  }

  void method_captures_this_with_guardian_refptr() {
    auto lambda = [this, protectedThis = RefPtr { &*this }]() {
      nonTrivial();
    };
    call(lambda);
  }


  void forEach(const WTF::Function<void(RefCountable&)>&);
  void method_captures_this_with_lambda_with_no_escape() {
    auto run = [&]([[clang::noescape]] const WTF::Function<void(RefCountable&)>& func) {
      forEach(func);
    };
    run([&](RefCountable&) {
      nonTrivial();
    });
  }
};

struct NonRefCountableWithLambdaCapturingThis {
  void nonTrivial();

  void method_captures_this_safe() {
    auto lambda = [&]() {
      nonTrivial();
    };
    lambda();
  }

  void method_captures_this_unsafe() {
    auto lambda = [&]() {
      nonTrivial();
    };
    call(lambda);
  }
};

void trivial_lambda() {
  RefCountable* ref_countable = make_obj();
  auto trivial_lambda = [&]() {
    return ref_countable->trivial();
  };
  trivial_lambda();
}

void lambda_with_args(RefCountable* obj) {
  auto trivial_lambda = [&](int v) {
    obj->method();
  };
  trivial_lambda(1);
}

void callFunctionOpaque(WTF::Function<void()>&&);
void callFunction(WTF::Function<void()>&& function) {
  someFunction();
  function();
}

void lambda_converted_to_function(RefCountable* obj)
{
  callFunction([&]() {
    obj->method();
    // expected-warning@-1{{Implicitly captured raw-pointer 'obj' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
  });
  callFunctionOpaque([&]() {
    obj->method();
    // expected-warning@-1{{Implicitly captured raw-pointer 'obj' to ref-counted type or CheckedPtr-capable type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
  });
}
