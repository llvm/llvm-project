// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.RefCntblBaseVirtualDtor -verify %s

struct RefCntblBase {
  void ref() {}
  void deref() {}
};

template<class T>
struct DerivedClassTmpl1 : T { };
// expected-warning@-1{{Struct 'RefCntblBase' is used as a base of struct 'DerivedClassTmpl1<RefCntblBase>' but doesn't have virtual destructor}}

DerivedClassTmpl1<RefCntblBase> a;
void foo(DerivedClassTmpl1<RefCntblBase>& obj) { obj.deref(); }

template<class T>
struct DerivedClassTmpl2 : T { };
// expected-warning@-1{{Struct 'RefCntblBase' is used as a base of struct 'DerivedClassTmpl2<RefCntblBase>' but doesn't have virtual destructor}}

template<class T> int foo(T) { DerivedClassTmpl2<T> f; return 42; }
int b = foo(RefCntblBase{});


template<class T>
struct DerivedClassTmpl3 : T { };
// expected-warning@-1{{Struct 'RefCntblBase' is used as a base of struct 'DerivedClassTmpl3<RefCntblBase>' but doesn't have virtual destructor}}

typedef DerivedClassTmpl3<RefCntblBase> Foo;
Foo c;

namespace WTF {

class RefCountedBase {
public:
  void ref() const { ++count; }

protected:
  bool derefBase() const
  {
    return !--count;
  }

private:
  mutable unsigned count;
};

template <typename T>
class RefCounted : public RefCountedBase {
public:
  void deref() const {
    if (derefBase())
      delete const_cast<T*>(static_cast<const T*>(this));
  }

protected:
  RefCounted() { }
};

template <typename X, typename T>
class ExoticRefCounted : public RefCountedBase {
public:
  void deref() const {
    if (derefBase())
      delete (const_cast<T*>(static_cast<const T*>(this)));
  }
};

template <typename X, typename T>
class BadBase : RefCountedBase {
public:
  void deref() const {
    if (derefBase())
      delete (const_cast<X*>(static_cast<const X*>(this)));
  }
};

template <typename T>
class FancyDeref {
public:
  void ref() const
  {
    ++refCount;
  }

  void deref() const
  {
    --refCount;
    if (refCount)
      return;
    auto deleteThis = [this] {
      delete static_cast<const T*>(this);
    };
    deleteThis();
  }
private:
  mutable unsigned refCount { 0 };
};

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
    explicit CallableWrapper(CallableType&& callable)
        : m_callable(WTFMove(callable)) { }
    CallableWrapper(const CallableWrapper&) = delete;
    CallableWrapper& operator=(const CallableWrapper&) = delete;
    Out call(In... in) final { return m_callable(in...); }
  private:
    CallableType m_callable;
  };

} // namespace Detail

template<typename> class Function;

template <typename Out, typename... In>
class Function<Out(In...)> {
public:
  using Impl = Detail::CallableWrapperBase<Out, In...>;

  Function() = default;

  template<typename CallableType>
  Function(CallableType&& callable)
      : m_callableWrapper(new Detail::CallableWrapper<CallableType, Out, In...>>(callable)) { }

  template<typename FunctionType>
  Function(FunctionType f)
      : m_callableWrapper(new Detail::CallableWrapper<FunctionType, Out, In...>>(f)) { }

  ~Function() {
  }

  Out operator()(In... in) const {
      ASSERT(m_callableWrapper);
      return m_callableWrapper->call(in...);
  }

  explicit operator bool() const { return !!m_callableWrapper; }

private:
  Impl* m_callableWrapper;
};

void ensureOnMainThread(const Function<void()>&& function);

enum class DestructionThread { Any, MainThread };

template <typename T, DestructionThread destructionThread = DestructionThread::Any>
class FancyDeref2 {
public:
  void ref() const
  {
    ++refCount;
  }

  void deref() const
  {
    --refCount;
    if (refCount)
      return;
    const_cast<FancyDeref2<T, destructionThread>*>(this)->destroy();
  }

private:
  void destroy() {
    delete static_cast<T*>(this);
  }
  mutable unsigned refCount { 0 };
};

template <typename S>
class DerivedFancyDeref2 : public FancyDeref2<S> {
};

template <typename T>
class BadFancyDeref {
public:
  void ref() const
  {
    ++refCount;
  }

  void deref() const
  {
    --refCount;
    if (refCount)
      return;
    auto deleteThis = [this] {
      delete static_cast<const T*>(this);
    };
    delete this;
  }
private:
  mutable unsigned refCount { 0 };
};

template <typename T>
class ThreadSafeRefCounted {
public:
  void ref() const { ++refCount; }
  void deref() const {
    if (!--refCount)
      delete const_cast<T*>(static_cast<const T*>(this));
  }
private:
  mutable unsigned refCount { 0 };
};

template <typename T>
class ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr {
public:
  void ref() const { ++refCount; }
  void deref() const {
    if (!--refCount)
      delete const_cast<T*>(static_cast<const T*>(this));
  }
private:
  mutable unsigned refCount { 0 };
};

} // namespace WTF

class DerivedClass4 : public WTF::RefCounted<DerivedClass4> { };

class DerivedClass4b : public WTF::ExoticRefCounted<int, DerivedClass4b> { };

class DerivedClass4cSub;
class DerivedClass4c : public WTF::BadBase<DerivedClass4cSub, DerivedClass4c> { };
// expected-warning@-1{{Class 'WTF::BadBase<DerivedClass4cSub, DerivedClass4c>' is used as a base of class 'DerivedClass4c' but doesn't have virtual destructor}}
class DerivedClass4cSub : public DerivedClass4c { };
void UseDerivedClass4c(DerivedClass4c &obj) { obj.deref(); }

class DerivedClass4d : public WTF::RefCounted<DerivedClass4d> {
public:
  virtual ~DerivedClass4d() { }
};
class DerivedClass4dSub : public DerivedClass4d { };

class DerivedClass5 : public DerivedClass4 { };
// expected-warning@-1{{Class 'DerivedClass4' is used as a base of class 'DerivedClass5' but doesn't have virtual destructor}}
void UseDerivedClass5(DerivedClass5 &obj) { obj.deref(); }

class DerivedClass6 : public WTF::ThreadSafeRefCounted<DerivedClass6> { };
void UseDerivedClass6(DerivedClass6 &obj) { obj.deref(); }

class DerivedClass7 : public DerivedClass6 { };
// expected-warning@-1{{Class 'DerivedClass6' is used as a base of class 'DerivedClass7' but doesn't have virtual destructor}}
void UseDerivedClass7(DerivedClass7 &obj) { obj.deref(); }

class DerivedClass8 : public WTF::ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<DerivedClass8> { };
void UseDerivedClass8(DerivedClass8 &obj) { obj.deref(); }

class DerivedClass9 : public DerivedClass8 { };
// expected-warning@-1{{Class 'DerivedClass8' is used as a base of class 'DerivedClass9' but doesn't have virtual destructor}}
void UseDerivedClass9(DerivedClass9 &obj) { obj.deref(); }

class DerivedClass10 : public WTF::FancyDeref<DerivedClass10> { };
void UseDerivedClass10(DerivedClass10 &obj) { obj.deref(); }

class DerivedClass10b : public WTF::DerivedFancyDeref2<DerivedClass10b> { };
void UseDerivedClass10b(DerivedClass10b &obj) { obj.deref(); }

class DerivedClass10c : public WTF::BadFancyDeref<DerivedClass10c> { };
// expected-warning@-1{{Class 'WTF::BadFancyDeref<DerivedClass10c>' is used as a base of class 'DerivedClass10c' but doesn't have virtual destructor}}
void UseDerivedClass10c(DerivedClass10c &obj) { obj.deref(); }

class BaseClass1 {
public:
  void ref() const { ++refCount; }
  void deref() const;
private:
  enum class Type { Base, Derived } type { Type::Base };
  mutable unsigned refCount { 0 };
};

class DerivedClass11 : public BaseClass1 { };

void BaseClass1::deref() const
{
  --refCount;
  if (refCount)
    return;
  switch (type) {
  case Type::Base:
    delete const_cast<BaseClass1*>(this);
    break;
  case Type::Derived:
    delete const_cast<DerivedClass11*>(static_cast<const DerivedClass11*>(this));
    break;
  }
}

void UseDerivedClass11(DerivedClass11& obj) { obj.deref(); }

class BaseClass2;
static void deleteBase2(BaseClass2*);

class BaseClass2 {
public:
  void ref() const { ++refCount; }
  void deref() const
  {
    if (!--refCount)
      deleteBase2(const_cast<BaseClass2*>(this));
  }
  virtual bool isDerived() { return false; }
private:
  mutable unsigned refCount { 0 };
};

class DerivedClass12 : public BaseClass2 {
  bool isDerived() final { return true; }
};

void UseDerivedClass11(DerivedClass12& obj) { obj.deref(); }

void deleteBase2(BaseClass2* obj) {
  if (obj->isDerived())
    delete static_cast<DerivedClass12*>(obj);
  else
    delete obj;
}

class BaseClass3 {
public:
  void ref() const { ++refCount; }
  void deref() const
  {
    if (!--refCount)
      const_cast<BaseClass3*>(this)->destory();
  }
  virtual bool isDerived() { return false; }

private:
  void destory();

  mutable unsigned refCount { 0 };
};

class DerivedClass13 : public BaseClass3 {
  bool isDerived() final { return true; }
};

void UseDerivedClass11(DerivedClass13& obj) { obj.deref(); }

void BaseClass3::destory() {
  if (isDerived())
    delete static_cast<DerivedClass13*>(this);
  else
    delete this;
}

class RecursiveBaseClass {
public:
  void ref() const {
    if (otherObject)
      otherObject->ref();
    else
      ++refCount;
  }
  void deref() const {
    if (otherObject)
      otherObject->deref();
    else {
      --refCount;
      if (refCount)
        return;
      delete this;
    }
  }
private:
  RecursiveBaseClass* otherObject { nullptr };
  mutable unsigned refCount { 0 };
};

class RecursiveDerivedClass : public RecursiveBaseClass { };
// expected-warning@-1{{Class 'RecursiveBaseClass' is used as a base of class 'RecursiveDerivedClass' but doesn't have virtual destructor}}

class DerivedClass14 : public WTF::RefCounted<DerivedClass14> {
public:
  virtual ~DerivedClass14() { }
};

void UseDerivedClass14(DerivedClass14& obj) { obj.deref(); }

class DerivedClass15 : public DerivedClass14 { };

void UseDerivedClass15(DerivedClass15& obj) { obj.deref(); }
