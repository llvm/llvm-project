// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoDeleteChecker -verify %s

#include "mock-types.h"

void someFunction();
void [[clang::annotate_type("webkit.nodelete")]] safeFunction();

void functionWithoutNoDeleteAnnotation() {
  someFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] callsUnsafe() {
  // expected-warning@-1{{A function 'callsUnsafe' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  someFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] callsSafe() {
  safeFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] declWithNoDelete();
void declWithNoDelete() {
  // expected-warning@-1{{A function 'declWithNoDelete' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  someFunction();
}

void defWithNoDelete();
void [[clang::annotate_type("webkit.nodelete")]] defWithNoDelete() {
// expected-warning@-1{{A function 'defWithNoDelete' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  someFunction();
}

class WeakRefCountable : public CanMakeWeakPtr<WeakRefCountable> {
public:
  static Ref<WeakRefCountable> create();

  ~WeakRefCountable();

  void ref() { m_refCount++; }
  void deref() {
    m_refCount--;
    if (!m_refCount)
      delete this;
  }

private:
  WeakRefCountable();

  unsigned m_refCount { 0 };
};

class SomeClass {
public:

  void [[clang::annotate_type("webkit.nodelete")]] someMethod();
  void [[clang::annotate_type("webkit.nodelete")]] unsafeMethod() {
    // expected-warning@-1{{A function 'unsafeMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    someFunction();
  }
  void [[clang::annotate_type("webkit.nodelete")]] safeMethod() {
    safeFunction();
  }

  virtual void [[clang::annotate_type("webkit.nodelete")]] someVirtualMethod();
  virtual void [[clang::annotate_type("webkit.nodelete")]] unsafeVirtualMethod() {
    // expected-warning@-1{{A function 'unsafeVirtualMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    someFunction();
  }
  virtual void [[clang::annotate_type("webkit.nodelete")]] safeVirtualMethod() {
    safeFunction();
  }

  static void [[clang::annotate_type("webkit.nodelete")]] someStaticMethod();
  static void [[clang::annotate_type("webkit.nodelete")]] unsafeStaticMethod() {
    // expected-warning@-1{{A function 'unsafeStaticMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    someFunction();
  }
  static void [[clang::annotate_type("webkit.nodelete")]] safeStaticMethod() {
    safeFunction();
  }

  virtual void [[clang::annotate_type("webkit.nodelete")]] anotherVirtualMethod();

  void [[clang::annotate_type("webkit.nodelete")]] setObj(RefCountable* obj) {
    // expected-warning@-1{{A function 'setObj' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    m_obj = obj;
  }

  void [[clang::annotate_type("webkit.nodelete")]] swapObj(RefPtr<RefCountable>&& obj) {
    m_obj.swap(obj);
  }

  void [[clang::annotate_type("webkit.nodelete")]] clearObj(RefCountable* obj) {
    // expected-warning@-1{{A function 'clearObj' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    m_obj = nullptr;
  }

  void [[clang::annotate_type("webkit.nodelete")]] deposeArg(WeakRefCountable&& unused) {
  }

  void [[clang::annotate_type("webkit.nodelete")]] deposeArgPtr(RefPtr<RefCountable>&& unused) {
  }

  enum class E : unsigned char { V1, V2 };
  bool [[clang::annotate_type("webkit.nodelete")]] deposeArgEnum() {
    E&& e = E::V1;
    return e != E::V2;
  }

  void [[clang::annotate_type("webkit.nodelete")]] deposeLocal() {
    // expected-warning@-1{{A function 'deposeLocal' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    RefPtr<RefCountable> obj = std::move(m_obj);
  }

  RefPtr<RefCountable> [[clang::annotate_type("webkit.nodelete")]] copyRefPtr() {
    return m_obj;
  }

  Ref<WeakRefCountable> [[clang::annotate_type("webkit.nodelete")]] copyRef() {
    return *m_weakObj.get();
  }

  RefPtr<WeakRefCountable> [[clang::annotate_type("webkit.nodelete")]] getWeakPtr() {
    return m_weakObj.get();
  }

  WeakRefCountable* [[clang::annotate_type("webkit.nodelete")]] useWeakPtr() {
    WeakPtr localWeak = m_weakObj.get();
    return localWeak.get();
  }

private:
  RefPtr<RefCountable> m_obj;
  Ref<RefCountable> m_ref;
  WeakPtr<WeakRefCountable> m_weakObj;
};

class IntermediateClass : public SomeClass {
  void anotherVirtualMethod() override;
};

class DerivedClass : public IntermediateClass {
  void anotherVirtualMethod() override {
    // expected-warning@-1{{A function 'anotherVirtualMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    someFunction();
  }
};

template <class Type>
class Base {
public:
  virtual unsigned foo() const = 0;
};

template <class Type>
class Derived : public Base<Type> {
public:
  virtual unsigned foo() const { return 0; }
};

struct Data {
  static Ref<Data> create() {
    return adoptRef(*new Data);
  }

  void ref() {
    ++refCount;
  }

  void deref() {
    --refCount;
    if (!refCount)
      delete this;
  }

  virtual void doSomething() { }

  int a[3] { 0 };

protected:
  Data() = default;

private:
  unsigned refCount { 0 };
};

struct SubData : Data {
  static Ref<SubData> create() {
    return adoptRef(*new SubData);
  }

  void doSomething() override { }

private:
  SubData() = default;
};

void [[clang::annotate_type("webkit.nodelete")]] makeData() {
  RefPtr<Data> constantData[2] = { Data::create() };
  RefPtr<Data> data[] = { Data::create() };
}

void [[clang::annotate_type("webkit.nodelete")]] makeSubData() {
  // expected-warning@-1{{A function 'makeSubData' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  SubData::create()->doSomething();
}
