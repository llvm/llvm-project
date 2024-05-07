// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.RefCntblBaseVirtualDtor -verify %s

struct RefCntblBase {
  void ref() {}
  void deref() {}
};

template<class T>
struct DerivedClassTmpl1 : T { };
// expected-warning@-1{{Struct 'RefCntblBase' is used as a base of struct 'DerivedClassTmpl1<RefCntblBase>' but doesn't have virtual destructor}}

DerivedClassTmpl1<RefCntblBase> a;



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

template <typename T>
class ThreadSafeRefCounted {
public:
  void ref() const;
  bool deref() const;
};

template <typename T>
class ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr {
public:
  void ref() const;
  bool deref() const;
};

} // namespace WTF

class DerivedClass4 : public WTF::RefCounted<DerivedClass4> { };

class DerivedClass5 : public DerivedClass4 { };
// expected-warning@-1{{Class 'DerivedClass4' is used as a base of class 'DerivedClass5' but doesn't have virtual destructor}}

class DerivedClass6 : public WTF::ThreadSafeRefCounted<DerivedClass6> { };

class DerivedClass7 : public DerivedClass6 { };
// expected-warning@-1{{Class 'DerivedClass6' is used as a base of class 'DerivedClass7' but doesn't have virtual destructor}}

class DerivedClass8 : public WTF::ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<DerivedClass8> { };

class DerivedClass9 : public DerivedClass8 { };
// expected-warning@-1{{Class 'DerivedClass8' is used as a base of class 'DerivedClass9' but doesn't have virtual destructor}}
