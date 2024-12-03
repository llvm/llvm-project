// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.UncountedLambdaCapturesChecker -verify %s

struct A {
  static void b();
};

struct RefCountable {
  void ref() {}
  void deref() {}
  void method();
  void constMethod() const;
  int trivial() { return 123; }
  RefCountable* next();
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
