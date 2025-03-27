// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedLambdaCapturesChecker -verify %s

#include "objc-mock-types.h"

namespace std {

template <typename T>
class unique_ptr {
private:
  T *t;

public:
  unique_ptr() : t(nullptr) { }
  unique_ptr(T *t) : t(t) { }
  ~unique_ptr() {
    if (t)
      delete t;
  }
  template <typename U> unique_ptr(unique_ptr<U>&& u)
    : t(u.t)
  {
    u.t = nullptr;
  }
  T *get() const { return t; }
  T *operator->() const { return t; }
  T &operator*() const { return *t; }
  unique_ptr &operator=(T *) { return *this; }
  explicit operator bool() const { return !!t; }
};

};

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

template <typename KeyType, typename ValueType>
class HashMap {
public:
  HashMap();
  HashMap([[clang::noescape]] const Function<ValueType()>&);
  void ensure(const KeyType&, [[clang::noescape]] const Function<ValueType()>&);
  bool operator+([[clang::noescape]] const Function<ValueType()>&) const;
  static void ifAny(HashMap, [[clang::noescape]] const Function<bool(ValueType)>&);

private:
  ValueType* m_table { nullptr };
};

} // namespace WTF

struct A {
  static void b();
};

SomeObj* make_obj();
CFMutableArrayRef make_cf();

void someFunction();
template <typename Callback> void call(Callback callback) {
  someFunction();
  callback();
}
void callAsync(const WTF::Function<void()>&);

void raw_ptr() {
  SomeObj* obj = make_obj();
  auto foo1 = [obj](){
    // expected-warning@-1{{Captured raw-pointer 'obj' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
    [obj doWork];
  };
  call(foo1);

  auto foo2 = [&obj](){
    // expected-warning@-1{{Captured raw-pointer 'obj' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
    [obj doWork];
  };
  auto foo3 = [&](){
    [obj doWork];
    // expected-warning@-1{{Implicitly captured raw-pointer 'obj' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
    obj = nullptr;
  };
  auto foo4 = [=](){
    [obj doWork];
    // expected-warning@-1{{Implicitly captured raw-pointer 'obj' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
  };
  
  auto cf = make_cf();
  auto bar1 = [cf](){
    // expected-warning@-1{{Captured reference 'cf' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
    CFArrayAppendValue(cf, nullptr);
  };
  auto bar2 = [&cf](){
    // expected-warning@-1{{Captured reference 'cf' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
    CFArrayAppendValue(cf, nullptr);
  };
  auto bar3 = [&](){
    CFArrayAppendValue(cf, nullptr);
    // expected-warning@-1{{Implicitly captured reference 'cf' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
    cf = nullptr;
  };
  auto bar4 = [=](){
    CFArrayAppendValue(cf, nullptr);
    // expected-warning@-1{{Implicitly captured reference 'cf' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
  };

  call(foo1);
  call(foo2);
  call(foo3);
  call(foo4);

  call(bar1);
  call(bar2);
  call(bar3);
  call(bar4);

  // Confirm that the checker respects [[clang::suppress]].
  SomeObj* suppressed_obj = nullptr;
  [[clang::suppress]] auto foo5 = [suppressed_obj](){
    [suppressed_obj doWork];
  };
  // no warning.
  call(foo5);

  // Confirm that the checker respects [[clang::suppress]].
  CFMutableArrayRef suppressed_cf = nullptr;
  [[clang::suppress]] auto bar5 = [suppressed_cf](){
    CFArrayAppendValue(suppressed_cf, nullptr);
  };
  // no warning.
  call(bar5);
}

void quiet() {
// This code is not expected to trigger any warnings.
  SomeObj *obj;

  auto foo3 = [&]() {};
  auto foo4 = [=]() {};

  call(foo3);
  call(foo4);

  obj = nullptr;
}

template <typename Callback>
void map(SomeObj* start, [[clang::noescape]] Callback&& callback)
{
  while (start) {
    callback(start);
    start = [start next];
  }
}

template <typename Callback1, typename Callback2>
void doubleMap(SomeObj* start, [[clang::noescape]] Callback1&& callback1, Callback2&& callback2)
{
  while (start) {
    callback1(start);
    callback2(start);
    start = [start next];
  }
}

template <typename Callback1, typename Callback2>
void get_count_cf(CFArrayRef array, [[clang::noescape]] Callback1&& callback1, Callback2&& callback2)
{
  auto count = CFArrayGetCount(array);
  callback1(count);
  callback2(count);
}

void noescape_lambda() {
  SomeObj* someObj = make_obj();
  SomeObj* otherObj = make_obj();
  map(make_obj(), [&](SomeObj *obj) {
    [otherObj doWork];
  });
  doubleMap(make_obj(), [&](SomeObj *obj) {
    [otherObj doWork];
  }, [&](SomeObj *obj) {
    [otherObj doWork];
    // expected-warning@-1{{Implicitly captured raw-pointer 'otherObj' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
  });
  ([&] {
    [someObj doWork];
  })();

  CFMutableArrayRef someCF = make_cf();
  get_count_cf(make_cf(), [&](CFIndex count) {
    CFArrayAppendValue(someCF, nullptr);
  }, [&](CFIndex count) {
    CFArrayAppendValue(someCF, nullptr);
    // expected-warning@-1{{Implicitly captured reference 'someCF' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
  });
}

void callFunctionOpaque(WTF::Function<void()>&&);
void callFunction(WTF::Function<void()>&& function) {
  someFunction();
  function();
}

void lambda_converted_to_function(SomeObj* obj, CFMutableArrayRef cf)
{
  callFunction([&]() {
    [obj doWork];
    // expected-warning@-1{{Implicitly captured raw-pointer 'obj' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
    CFArrayAppendValue(cf, nullptr);
    // expected-warning@-1{{Implicitly captured reference 'cf' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
  });
  callFunctionOpaque([&]() {
    [obj doWork];
    // expected-warning@-1{{Implicitly captured raw-pointer 'obj' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
    CFArrayAppendValue(cf, nullptr);
    // expected-warning@-1{{Implicitly captured reference 'cf' to unretained type is unsafe [alpha.webkit.UnretainedLambdaCapturesChecker]}}
  });
}

@interface ObjWithSelf : NSObject {
  RetainPtr<id> delegate;
}
-(void)doWork;
-(void)run;
@end

@implementation ObjWithSelf
-(void)doWork {
  auto doWork = [&] {
    someFunction();
    [delegate doWork];
  };
  callFunctionOpaque(doWork);
}
-(void)run {
  someFunction();
}
@end