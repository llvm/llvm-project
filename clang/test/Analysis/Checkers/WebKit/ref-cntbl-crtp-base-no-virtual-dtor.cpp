// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.RefCntblBaseVirtualDtor -verify %s

#include "mock-types.h"

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
    Out call(In... in) final;
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
    Function(FunctionType f);

    Out operator()(In... in) const;
    explicit operator bool() const { return !!m_callableWrapper; }

private:
    enum AdoptTag { Adopt };
    Function(Impl* impl, AdoptTag)
        : m_callableWrapper(impl)
    {
    }

    friend Function adopt<Out, In...>(Impl*);

    Impl* m_callableWrapper;
};

template<typename Out, typename... In> Function<Out(In...)> adopt(Detail::CallableWrapperBase<Out, In...>* impl)
{
    return Function<Out(In...)>(impl, Function<Out(In...)>::Adopt);
}

enum class DestructionThread : unsigned char { Any, Main, MainRunLoop };
void ensureOnMainThread(Function<void()>&&); // Sync if called on main thread, async otherwise.
void ensureOnMainRunLoop(Function<void()>&&); // Sync if called on main run loop, async otherwise.

class ThreadSafeRefCountedBase {
public:
    ThreadSafeRefCountedBase() = default;

    void ref() const
    {
        ++m_refCount;
    }

    bool hasOneRef() const
    {
        return refCount() == 1;
    }

    unsigned refCount() const
    {
        return m_refCount;
    }

protected:
    bool derefBase() const
    {
      if (!--m_refCount) {
          m_refCount = 1;
          return true;
      }
      return false;
    }

private:
    mutable unsigned m_refCount { 1 };
};

template<class T, DestructionThread destructionThread = DestructionThread::Any> class ThreadSafeRefCounted : public ThreadSafeRefCountedBase {
public:
    void deref() const
    {
        if (!derefBase())
            return;

        if constexpr (destructionThread == DestructionThread::Any) {
            delete static_cast<const T*>(this);
        } else if constexpr (destructionThread == DestructionThread::Main) {
            ensureOnMainThread([this] {
                delete static_cast<const T*>(this);
            });
        } else if constexpr (destructionThread == DestructionThread::MainRunLoop) {
            auto deleteThis = [this] {
                delete static_cast<const T*>(this);
            };
            ensureOnMainThread(deleteThis);
        }
    }

protected:
    ThreadSafeRefCounted() = default;
};

class FancyRefCountedClass final : public ThreadSafeRefCounted<FancyRefCountedClass, DestructionThread::Main> {
public:
    static Ref<FancyRefCountedClass> create()
    {
        return adoptRef(*new FancyRefCountedClass());
    }

    virtual ~FancyRefCountedClass();

private:
    FancyRefCountedClass();
};

template<class T, DestructionThread destructionThread = DestructionThread::Any> class BadThreadSafeRefCounted : public ThreadSafeRefCountedBase {
public:
    void deref() const
    {
        if (!derefBase())
            return;

        [this] {
          delete static_cast<const T*>(this);
        };
    }

protected:
    BadThreadSafeRefCounted() = default;
};

class FancyRefCountedClass2 final : public ThreadSafeRefCounted<FancyRefCountedClass, DestructionThread::Main> {
// expected-warning@-1{{Class 'ThreadSafeRefCounted<FancyRefCountedClass, DestructionThread::Main>' is used as a base of class 'FancyRefCountedClass2' but doesn't have virtual destructor}}
public:
    static Ref<FancyRefCountedClass2> create()
    {
        return adoptRef(*new FancyRefCountedClass2());
    }

    virtual ~FancyRefCountedClass2();

private:
    FancyRefCountedClass2();
};

template<class T, DestructionThread destructionThread = DestructionThread::Any> class NestedThreadSafeRefCounted : public ThreadSafeRefCountedBase {
public:
    void deref() const
    {
        if (!derefBase())
            return;
        ensureOnMainRunLoop([&] {
          auto destroyThis = [&] {
            delete static_cast<const T*>(this);
          };
          destroyThis();
        });
    }

protected:
    NestedThreadSafeRefCounted() = default;
};

class FancyRefCountedClass3 final : public NestedThreadSafeRefCounted<FancyRefCountedClass3, DestructionThread::Main> {
public:
    static Ref<FancyRefCountedClass3> create()
    {
        return adoptRef(*new FancyRefCountedClass3());
    }

    virtual ~FancyRefCountedClass3();

private:
    FancyRefCountedClass3();
};

template<class T, DestructionThread destructionThread = DestructionThread::Any> class BadNestedThreadSafeRefCounted : public ThreadSafeRefCountedBase {
public:
    void deref() const
    {
        if (!derefBase())
            return;
        ensureOnMainThread([&] {
          auto destroyThis = [&] {
            delete static_cast<const T*>(this);
          };
        });
    }

protected:
    BadNestedThreadSafeRefCounted() = default;
};

class FancyRefCountedClass4 final : public BadNestedThreadSafeRefCounted<FancyRefCountedClass4, DestructionThread::Main> {
// expected-warning@-1{{Class 'BadNestedThreadSafeRefCounted<FancyRefCountedClass4, DestructionThread::Main>' is used as a base of class 'FancyRefCountedClass4' but doesn't have virtual destructor}}
public:
    static Ref<FancyRefCountedClass4> create()
    {
        return adoptRef(*new FancyRefCountedClass4());
    }

    virtual ~FancyRefCountedClass4();

private:
    FancyRefCountedClass4();
};

class FancyRefCountedClass5 final : public ThreadSafeRefCounted<FancyRefCountedClass5, DestructionThread::MainRunLoop> {
public:
    static Ref<FancyRefCountedClass5> create()
    {
        return adoptRef(*new FancyRefCountedClass5());
    }

    virtual ~FancyRefCountedClass5();

private:
    FancyRefCountedClass5();
};
