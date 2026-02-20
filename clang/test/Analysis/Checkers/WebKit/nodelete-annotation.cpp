// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoDeleteChecker -verify %s

#include "mock-types.h"

void someFunction();
RefCountable* [[clang::annotate_type("webkit.nodelete")]] safeFunction();

void functionWithoutNoDeleteAnnotation() {
  someFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] callsUnsafe() {
  someFunction(); // expected-warning{{A function 'callsUnsafe' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] callsSafe() {
  safeFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] declWithNoDelete();
void declWithNoDelete() {
  someFunction(); // expected-warning{{A function 'declWithNoDelete' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}
void defWithNoDelete();
void [[clang::annotate_type("webkit.nodelete")]] defWithNoDelete() {
  someFunction(); // expected-warning{{A function 'defWithNoDelete' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] funncWithUnsafeParam(Ref<RefCountable> t) {
  // expected-warning@-1{{A function 'funncWithUnsafeParam' has [[clang::annotate_type("webkit.nodelete")]] but it contains a parameter 't' which could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] funncWithUnsafeParam(unsigned safe, Ref<RefCountable> unsafe) {
  // expected-warning@-1{{A function 'funncWithUnsafeParam' has [[clang::annotate_type("webkit.nodelete")]] but it contains a parameter 'unsafe' which could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] funncWithUnsafeParam(Ref<RefCountable> unsafe, unsigned safe) {
  // expected-warning@-1{{A function 'funncWithUnsafeParam' has [[clang::annotate_type("webkit.nodelete")]] but it contains a parameter 'unsafe' which could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] funncWithSafeParam(Ref<RefCountable>& safe1, Ref<RefCountable>* safe2) {
}

void [[clang::annotate_type("webkit.nodelete")]] callsUnsafeInDoWhile() {
  do {
    someFunction(); // expected-warning{{A function 'callsUnsafeInDoWhile' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  } while(0);
}

void [[clang::annotate_type("webkit.nodelete")]] callsUnsafeInIf(bool safe) {
  if (safe)
    safeFunction();
  else
    someFunction(); // expected-warning{{A function 'callsUnsafeInIf' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] declaresUnsafeVar(bool safe) {
  if (safe) {
    auto* t = safeFunction();
  } else {
    RefPtr<RefCountable> t = safeFunction();
    // expected-warning@-1{{A function 'declaresUnsafeVar' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
}

void [[clang::annotate_type("webkit.nodelete")]] declaresVarInIf(bool safe) {
  if (RefPtr<RefCountable> t = safeFunction()) {
    // expected-warning@-1{{A function 'declaresVarInIf' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    t->method();
  }
}

template <typename T>
struct TemplatedClass {
  void [[clang::annotate_type("webkit.nodelete")]] methodCallsUnsafe(T* t) {
    t->method(); // expected-warning{{A function 'methodCallsUnsafe' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
  void [[clang::annotate_type("webkit.nodelete")]] methodCallsSafe(T* t) {
    t->trivial();
  }
};

using TemplatedToRefCountable = TemplatedClass<RefCountable>;
void useTemplatedToRefCountable() {
  TemplatedToRefCountable c;
  c.methodCallsUnsafe(nullptr);
  c.methodCallsSafe(nullptr);
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
    someFunction(); // expected-warning{{A function 'unsafeMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
  void [[clang::annotate_type("webkit.nodelete")]] safeMethod() {
    safeFunction();
  }

  virtual void [[clang::annotate_type("webkit.nodelete")]] someVirtualMethod();
  virtual void [[clang::annotate_type("webkit.nodelete")]] unsafeVirtualMethod() {
    someFunction(); // expected-warning{{A function 'unsafeVirtualMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
  virtual void [[clang::annotate_type("webkit.nodelete")]] safeVirtualMethod() {
    safeFunction();
  }

  static void [[clang::annotate_type("webkit.nodelete")]] someStaticMethod();
  static void [[clang::annotate_type("webkit.nodelete")]] unsafeStaticMethod() {
    someFunction(); // expected-warning{{A function 'unsafeStaticMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
  static void [[clang::annotate_type("webkit.nodelete")]] safeStaticMethod() {
    safeFunction();
  }

  virtual void [[clang::annotate_type("webkit.nodelete")]] anotherVirtualMethod();

  void [[clang::annotate_type("webkit.nodelete")]] setObj(RefCountable* obj) {
    m_obj = obj; // expected-warning{{A function 'setObj' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] swapObj(RefPtr<RefCountable>&& obj) {
    m_obj.swap(obj);
  }

  void [[clang::annotate_type("webkit.nodelete")]] clearObj(RefCountable* obj) {
    m_obj = nullptr; // expected-warning{{A function 'clearObj' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
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
    RefPtr<RefCountable> obj = std::move(m_obj); // expected-warning{{A function 'deposeLocal' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
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
    someFunction(); // expected-warning{{A function 'anotherVirtualMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
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
  SubData::create()->doSomething();
  // expected-warning@-1{{A function 'makeSubData' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}
