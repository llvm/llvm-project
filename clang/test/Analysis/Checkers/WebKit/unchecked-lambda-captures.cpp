// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedLambdaCapturesChecker -verify %s

#include "mock-types.h"

struct A {
  static void b();
};

CheckedObj* make_obj();

void someFunction();
template <typename Callback> void call(Callback callback) {
  someFunction();
  callback();
}
void callAsync(const WTF::Function<void()>&);

void raw_ptr() {
  CheckedObj* checked_ptr_capable = make_obj();
  auto foo1 = [checked_ptr_capable](){
    // expected-warning@-1{{Captured variable 'checked_ptr_capable' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
    checked_ptr_capable->method();
  };
  auto foo2 = [&checked_ptr_capable](){
    // expected-warning@-1{{Captured variable 'checked_ptr_capable' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
    checked_ptr_capable->method();
  };
  auto foo3 = [&](){
    checked_ptr_capable->method();
    // expected-warning@-1{{Implicitly captured variable 'checked_ptr_capable' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
    checked_ptr_capable = nullptr;
  };

  auto foo4 = [=](){
    checked_ptr_capable->method();
    // expected-warning@-1{{Implicitly captured variable 'checked_ptr_capable' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  };

  call(foo1);
  call(foo2);
  call(foo3);
  call(foo4);

  // Confirm that the checker respects [[clang::suppress]].
  CheckedObj* suppressed_checked_ptr_capable = nullptr;
  [[clang::suppress]] auto foo5 = [suppressed_checked_ptr_capable](){};
  // no warning.
  call(foo5);
}

void references() {
  CheckedObj automatic;
  CheckedObj& checked_ptr_capable_ref = automatic;
  auto foo1 = [checked_ptr_capable_ref](){ checked_ptr_capable_ref.constMethod(); };
  auto foo2 = [&checked_ptr_capable_ref](){ checked_ptr_capable_ref.method(); };
  // expected-warning@-1{{Captured variable 'checked_ptr_capable_ref' is a raw reference to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  auto foo3 = [&](){ checked_ptr_capable_ref.method(); };
  // expected-warning@-1{{Implicitly captured variable 'checked_ptr_capable_ref' is a raw reference to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  auto foo4 = [=](){ checked_ptr_capable_ref.constMethod(); };

  call(foo1);
  call(foo2);
  call(foo3);
  call(foo4);
}

void quiet() {
// This code is not expected to trigger any warnings.
  {
    CheckedObj automatic;
    CheckedObj &checked_ptr_capable_ref = automatic;
  }

  auto foo3 = [&]() {};
  auto foo4 = [=]() {};

  call(foo3);
  call(foo4);

  CheckedObj *checked_ptr_capable = nullptr;
}

template <typename Callback>
void map(CheckedObj* start, [[clang::noescape]] Callback&& callback)
{
  while (start) {
    callback(*start);
    start = start->next();
  }
}

template <typename Callback1, typename Callback2>
void doubleMap(CheckedObj* start, [[clang::noescape]] Callback1&& callback1, Callback2&& callback2)
{
  while (start) {
    callback1(*start);
    callback2(*start);
    start = start->next();
  }
}

void noescape_lambda() {
  CheckedObj* someObj = make_obj();
  CheckedObj* otherObj = make_obj();
  map(make_obj(), [&](CheckedObj& obj) {
    otherObj->method();
  });
  doubleMap(make_obj(), [&](CheckedObj& obj) {
    otherObj->method();
  }, [&](CheckedObj& obj) {
    otherObj->method();
    // expected-warning@-1{{Implicitly captured variable 'otherObj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
  ([&] {
    someObj->method();
  })();
}

void lambda_capture_param(CheckedObj* obj) {
  auto someLambda = [&]() {
    obj->method();
  };
  someLambda();
  someLambda();
}

struct CheckedObjWithLambdaCapturingThis {
  void incrementCheckedPtrCount() const;
  void decrementCheckedPtrCount() const;
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
      // expected-warning@-1{{Implicitly captured variable 'this' is a raw pointer to CheckedPtr-capable type 'CheckedObjWithLambdaCapturingThis' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
    };
    call(lambda);
  }

  void method_captures_this_unsafe_capture_local_var_explicitly() {
    CheckedObj* x = make_obj();
    call([this, protectedThis = CheckedPtr { this }, x]() {
      // expected-warning@-1{{Captured variable 'x' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
      nonTrivial();
      x->method();
    });
  }

  void method_captures_this_with_other_protected_var() {
    CheckedObj* x = make_obj();
    call([this, protectedX = CheckedPtr { x }]() {
      // expected-warning@-1{{Captured variable 'this' is a raw pointer to CheckedPtr-capable type 'CheckedObjWithLambdaCapturingThis' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
      nonTrivial();
      protectedX->method();
    });
  }

  void method_captures_this_unsafe_capture_local_var_explicitly_with_deref() {
    CheckedObj* x = make_obj();
    call([this, protectedThis = CheckedRef { *this }, x]() {
      // expected-warning@-1{{Captured variable 'x' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
      nonTrivial();
      x->method();
    });
  }

  void method_captures_this_unsafe_local_var_via_vardecl() {
    CheckedObj* x = make_obj();
    auto lambda = [this, protectedThis = CheckedRef { *this }, x]() {
      // expected-warning@-1{{Captured variable 'x' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
      nonTrivial();
      x->method();
    };
    call(lambda);
  }

  void method_captures_this_with_guardian() {
    auto lambda = [this, protectedThis = CheckedRef { *this }]() {
      nonTrivial();
    };
    call(lambda);
  }

  void method_captures_this_with_guardian_refptr() {
    auto lambda = [this, protectedThis = CheckedPtr { &*this }]() {
      nonTrivial();
    };
    call(lambda);
  }

  void forEach(const WTF::Function<void(CheckedObj&)>&);
  void method_captures_this_with_lambda_with_no_escape() {
    auto run = [&]([[clang::noescape]] const WTF::Function<void(CheckedObj&)>& func) {
      forEach(func);
    };
    run([&](CheckedObj&) {
      nonTrivial();
    });
  }

  static void callLambda([[clang::noescape]] const WTF::Function<CheckedPtr<CheckedObj>()>&);
  void method_captures_this_in_template_method() {
    CheckedObj* obj = make_obj();
    WTF::HashMap<int, CheckedPtr<CheckedObj>> nextMap;
    nextMap.ensure(3, [&] {
      return obj->next();
    });
    nextMap+[&] {
      return obj->next();
    };
    WTF::HashMap<int, CheckedPtr<CheckedObj>>::ifAny(nextMap, [&](auto& item) -> bool {
      return item->next() && obj->next();
    });
    callLambda([&]() -> CheckedPtr<CheckedObj> {
      return obj->next();
    });
    WTF::HashMap<int, CheckedPtr<CheckedObj>> anotherMap([&] {
      return obj->next();
    });
  }

  void callAsyncNoescape([[clang::noescape]] WTF::Function<bool(CheckedObj&)>&&);
  void method_temp_lambda(CheckedObj* obj) {
    callAsyncNoescape([this, otherObj = CheckedPtr { obj }](auto& obj) {
      return otherObj.get() == &obj;
    });
  }

  void method_nested_lambda() {
    callAsync([this, protectedThis = CheckedRef { *this }] {
      callAsync([this, protectedThis = static_cast<const CheckedRef<CheckedObjWithLambdaCapturingThis>&&>(protectedThis)] {
        nonTrivial();
      });
    });
  }

  void method_nested_lambda2() {
    callAsync([this, protectedThis = CheckedPtr { this }] {
      callAsync([this, protectedThis = std::move(*protectedThis)] {
        nonTrivial();
      });
    });
  }

  void method_nested_lambda3() {
    callAsync([this, protectedThis = CheckedPtr { this }] {
      callAsync([this] {
        // expected-warning@-1{{Captured variable 'this' is a raw pointer to CheckedPtr-capable type 'CheckedObjWithLambdaCapturingThis' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
        nonTrivial();
      });
    });
  }

  void method_nested_lambda4() {
    callAsync([this, protectedThis = CheckedPtr { this }] {
      callAsync([this, protectedThis = WTF::move(*protectedThis)] {
        nonTrivial();
      });
    });
  }
};

struct NonCheckedObjWithLambdaCapturingThis {
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
  CheckedObj* checked_ptr_capable = make_obj();
  auto trivial_lambda = [&]() {
    return checked_ptr_capable->trivial();
  };
  trivial_lambda();
}

bool call_lambda_var_decl() {
  CheckedObj* checked_ptr_capable = make_obj();
  auto lambda1 = [&]() -> bool {
    return checked_ptr_capable->next();
  };
  auto lambda2 = [=]() -> bool {
    return checked_ptr_capable->next();
  };
  return lambda1() && lambda2();
}

void lambda_with_args(CheckedObj* obj) {
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

void lambda_converted_to_function(CheckedObj* obj)
{
  callFunction([&]() {
    obj->method();
    // expected-warning@-1{{Implicitly captured variable 'obj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
  callFunctionOpaque([&]() {
    obj->method();
    // expected-warning@-1{{Implicitly captured variable 'obj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
}

void capture_copy_in_lambda(CheckedObj& checked) {
  callFunctionOpaque([checked]() mutable {
    checked.method();
  });
  auto* ptr = &checked;
  callFunctionOpaque([ptr]() mutable {
    // expected-warning@-1{{Captured variable 'ptr' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
    ptr->method();
  });
}

struct TemplateFunctionCallsLambda {
  void ref() const;
  void deref() const;

  CheckedObj* obj();

  template <typename T>
  CheckedPtr<T> method(T* t) {
    auto ret = ([&]() -> CheckedPtr<T> {
      if constexpr (T::isEncodable)
        return t;
      return obj() ? t : nullptr;
    })();
    return ret;
  }
};

class Iterator {
public:
  Iterator(void* array, unsigned long sizeOfElement, unsigned int index);
  Iterator(const Iterator&);
  Iterator& operator=(const Iterator&);
  bool operator==(const Iterator&);

  Iterator& operator++();
  int& operator*();

private:
  void* current { nullptr };
  unsigned long sizeOfElement { 0 };
};

void ranges_for_each(CheckedObj* obj) {
  int array[] = { 1, 2, 3, 4, 5 };
  std::ranges::for_each(Iterator(array, sizeof(*array), 0), Iterator(array, sizeof(*array), 5), [&](int& item) {
    obj->method();
    ++item;
  });
}

class IntCollection {
public:
  int* begin();
  int* end();
  const int* begin() const;
  const int* end() const;
};

class CheckedPtrCapableObj {
public:
  void incrementCheckedPtrCount();
  void decrementCheckedPtrCount();

  bool allOf(const IntCollection&);
  bool isMatch(int);

  void call() const;
  void callLambda([[clang::noescape]] const WTF::Function<void ()>& callback) const;
  void doSomeWork() const;
};

bool CheckedPtrCapableObj::allOf(const IntCollection& collection) {
  return std::ranges::all_of(collection, [&](auto& number) {
    return isMatch(number);
  });
}

void CheckedPtrCapableObj::callLambda([[clang::noescape]] const WTF::Function<void ()>& callback) const
{
    callback();
}

void CheckedPtrCapableObj::call() const
{
    auto lambda = [&] {
        doSomeWork();
    };
    callLambda(lambda);
}

void scope_exit(CheckedObj* obj) {
  auto scope = WTF::makeScopeExit([&] {
    obj->method();
  });
  someFunction();
  WTF::ScopeExit scope2([&] {
    obj->method();
  });
  someFunction();
}

void doWhateverWith(WTF::ScopeExit& obj);

void scope_exit_with_side_effect(CheckedObj* obj) {
  auto scope = WTF::makeScopeExit([&] {
    obj->method();
    // expected-warning@-1{{Implicitly captured variable 'obj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
  doWhateverWith(scope);
}

void scope_exit_static(CheckedObj* obj) {
  static auto scope = WTF::makeScopeExit([&] {
    obj->method();
    // expected-warning@-1{{Implicitly captured variable 'obj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
}

WTF::Function<void()> scope_exit_take_lambda(CheckedObj* obj) {
  auto scope = WTF::makeScopeExit([&] {
    obj->method();
    // expected-warning@-1{{Implicitly captured variable 'obj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
  return scope.take();
}

// FIXME: Ideally, we treat release() as a trivial function.
void scope_exit_release(CheckedObj* obj) {
  auto scope = WTF::makeScopeExit([&] {
    obj->method();
    // expected-warning@-1{{Implicitly captured variable 'obj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
  scope.release();
}

void make_visitor(CheckedObj* obj) {
  auto visitor = WTF::makeVisitor([&] {
    obj->method();
  });
}

void use_visitor(CheckedObj* obj) {
  auto visitor = WTF::makeVisitor([&] {
    obj->method();
  });
  WTF::visit(visitor, obj);
}

template <typename Visitor, typename ObjectType>
void bad_visit(Visitor&, ObjectType*) {
  someFunction();
}

void static_visitor(CheckedObj* obj) {
  static auto visitor = WTF::makeVisitor([&] {
    obj->method();
    // expected-warning@-1{{Implicitly captured variable 'obj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
}

void make_visitor_with_multiple_lambdas(CheckedObj* obj) {
  auto* otherObj = make_obj();
  auto visitor = WTF::makeVisitor([&] {
    obj->method();
    // expected-warning@-1{{Implicitly captured variable 'obj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  }, [&] {
    otherObj->method();
    // expected-warning@-1{{Implicitly captured variable 'otherObj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
  bad_visit(visitor, obj);
}

void bad_use_visitor(CheckedObj* obj) {
  auto visitor = WTF::makeVisitor([&] {
    obj->method();
    // expected-warning@-1{{Implicitly captured variable 'obj' is a raw pointer to CheckedPtr-capable type 'CheckedObj' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
  });
  bad_visit(visitor, obj);
}

class LambdaInConstructorDestructor {
public:
  LambdaInConstructorDestructor() {
    call([this]() {
      // expected-warning@-1{{Captured variable 'this' is a raw pointer to CheckedPtr-capable type 'LambdaInConstructorDestructor' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
      doWork();
    });
  }

  ~LambdaInConstructorDestructor() {
    call([this]() {
      // expected-warning@-1{{Captured variable 'this' is a raw pointer to CheckedPtr-capable type 'LambdaInConstructorDestructor' [alpha.webkit.UncheckedLambdaCapturesChecker]}}
      doWork();
    });
  }

  void incrementCheckedPtrCount() const;
  void decrementCheckedPtrCount() const;

  void doWork();
};
