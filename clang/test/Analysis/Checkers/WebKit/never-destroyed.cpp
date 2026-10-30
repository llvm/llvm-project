// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker,alpha.webkit.UncountedLocalVarsChecker,alpha.webkit.UncheckedCallArgsChecker,alpha.webkit.UncheckedLocalVarsChecker -verify %s

#include "mock-types.h"

namespace WTF {

template <typename T> class NeverDestroyed {
public:
  template <typename... Args> NeverDestroyed(Args&&...);
  operator T&() { return *storagePointer(); }
  T& get() { return *storagePointer(); }
  T* operator->() { return storagePointer(); }
  operator const T&() const { return *storagePointer(); }
  const T& get() const { return *storagePointer(); }
  const T* operator->() const { return storagePointer(); }

private:
  T* storagePointer() const { return reinterpret_cast<T*>(const_cast<char*>(m_storage)); }
  alignas(T) char m_storage[sizeof(T)];
};

template <typename T> class LazyNeverDestroyed {
public:
  template <typename... Args> void construct(Args&&...);
  operator T&() { return *storagePointer(); }
  T& get() { return *storagePointer(); }
  T* operator->() { return storagePointer(); }

private:
  T* storagePointer() const { return reinterpret_cast<T*>(const_cast<char*>(m_storage)); }
  alignas(T) char m_storage[sizeof(T)];
};

} // namespace WTF

using WTF::LazyNeverDestroyed;
using WTF::NeverDestroyed;

void consume(RefCountable*);
void consumeRef(RefCountable&);
void consume(CheckedObj*);
void consumeRef(CheckedObj&);

namespace static_local {

RefCountable& shared() {
  static NeverDestroyed<RefCountable> instance;
  instance->method();
  instance.get().method();
  consume(&instance.get());
  consumeRef(instance);
  RefCountable& ref = instance;
  ref.method();
  RefCountable* ptr = &instance.get();
  ptr->method();
  return instance;
}

CheckedObj& sharedChecked() {
  static NeverDestroyed<CheckedObj> instance;
  instance->method();
  consumeRef(instance);
  CheckedObj& ref = instance.get();
  ref.method();
  return instance;
}

void lazy() {
  static LazyNeverDestroyed<RefCountable> instance;
  instance.construct();
  instance->method();
  RefCountable& ref = instance;
  ref.method();
}

void constRef() {
  static NeverDestroyed<const Ref<RefCountable>> instance(RefCountable::create());
  instance->get().method();
  instance.get()->method();
  consume(instance.get().ptr());
  RefCountable* ptr = instance.get().ptr();
  ptr->method();
  RefCountable& ref = instance.get();
  ref.method();
}

void constRefPtr() {
  static NeverDestroyed<const RefPtr<RefCountable>> instance(RefCountable::create());
  instance.get()->method();
  consume(instance.get().get());
  RefCountable* ptr = instance.get().get();
  ptr->method();
}

void ref() {
  static NeverDestroyed<Ref<RefCountable>> instance(RefCountable::create());
  instance.get()->method();
  // expected-warning@-1{{Function argument 'instance.get()' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  RefCountable* ptr = instance.get().ptr();
  // expected-warning@-1{{Local variable 'ptr' is a raw pointer to RefPtr-capable type 'RefCountable'}}
  ptr->method();
}

void refPtr() {
  static NeverDestroyed<RefPtr<RefCountable>> instance(RefCountable::create());
  instance.get()->method();
  // expected-warning@-1{{Function argument 'instance.get()' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  RefCountable* ptr = instance.get().get();
  // expected-warning@-1{{Local variable 'ptr' is a raw pointer to RefPtr-capable type 'RefCountable'}}
  ptr->method();
}

void rawPointer() {
  static NeverDestroyed<RefCountable*> instance;
  instance.get()->method();
  // expected-warning@-1{{Function argument 'instance.get()' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  RefCountable* ptr = instance.get();
  // expected-warning@-1{{Local variable 'ptr' is a raw pointer to RefPtr-capable type 'RefCountable'}}
  ptr->method();
}

} // namespace static_local

namespace global {

NeverDestroyed<RefCountable> globalInstance;
LazyNeverDestroyed<CheckedObj> lazyGlobalInstance;

void foo() {
  globalInstance->method();
  RefCountable& ref = globalInstance;
  ref.method();
  lazyGlobalInstance->method();
  CheckedObj* ptr = &lazyGlobalInstance.get();
  ptr->method();
}

} // namespace global

namespace no_destroy {

[[clang::no_destroy]] static RefCountable refCountable;
[[clang::no_destroy]] static CheckedObj checkedObj;
[[clang::no_destroy]] static RefCountable* rawPointer;
[[clang::no_destroy]] static const RefPtr<RefCountable> constRefPtr;
[[clang::no_destroy]] static RefPtr<RefCountable> refPtr;

void foo() {
  refCountable.method();
  consume(&refCountable);
  RefCountable& ref = refCountable;
  ref.method();
  checkedObj.method();
  CheckedObj* ptr = &checkedObj;
  ptr->method();
}

void bar() {
  rawPointer->method();
  // expected-warning@-1{{Function argument 'rawPointer' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  RefCountable* ptr = rawPointer;
  // expected-warning@-1{{Local variable 'ptr' is a raw pointer to RefPtr-capable type 'RefCountable'}}
  ptr->method();
}

void constSmartPointer() {
  constRefPtr->method();
  RefCountable* ptr = constRefPtr.get();
  ptr->method();
}

void smartPointer() {
  RefCountable* ptr = refPtr.get();
  // expected-warning@-1{{Local variable 'ptr' is a raw pointer to RefPtr-capable type 'RefCountable'}}
  ptr->method();
}

} // namespace no_destroy

namespace not_immortal {

RefCountable* provide();

void foo() {
  RefCountable* ptr = provide();
  // expected-warning@-1{{Local variable 'ptr' is a raw pointer to RefPtr-capable type 'RefCountable'}}
  ptr->method();
}

} // namespace not_immortal
