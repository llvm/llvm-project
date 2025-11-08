#ifndef std_move
#define std_move

namespace std {

template <typename T> struct remove_reference {
typedef T type;
};

template <typename T> struct remove_reference<T&> {
typedef T type;
};

template<typename T> typename remove_reference<T>::type&& move(T&& t);

}

#endif

#ifndef mock_types_1103988513531
#define mock_types_1103988513531

namespace std {

template <typename T>
class unique_ptr {
private:
  void *t;

public:
  unique_ptr() : t(nullptr) { }
  unique_ptr(T *t) : t(t) { }
  ~unique_ptr() {
    if (t)
      delete static_cast<T*>(t);
  }
  template <typename U> unique_ptr(unique_ptr<U>&& u)
    : t(u.t)
  {
    u.t = nullptr;
  }
  T *get() const { return static_cast<T*>(t); }
  T *operator->() const { return get(); }
  T &operator*() const { return *get(); }
  unique_ptr &operator=(T *) { return *this; }
  explicit operator bool() const { return !!t; }
};

};

template<typename T>
struct RawPtrTraits {
  using StorageType = T*;

  template<typename U>
  static T* exchange(StorageType& ptr, U&& newValue)
  {
    StorageType oldValue = static_cast<StorageType&&>(ptr);
    ptr = static_cast<U&&>(newValue);
    return oldValue;
  }

  static void swap(StorageType& a, StorageType& b)
  {
    StorageType temp = static_cast<StorageType&&>(a);
    a = static_cast<StorageType&&>(b);
    b = static_cast<StorageType&&>(temp);
  }
  static T* unwrap(const StorageType& ptr) { return ptr; }
};

template<typename T> struct DefaultRefDerefTraits {
  static T* refIfNotNull(T* ptr)
  {
    if (ptr)
      ptr->ref();
    return ptr;
  }

  static T& ref(T& ref)
  {
    ref.ref();
    return ref;
  }

  static void derefIfNotNull(T* ptr)
  {
    if (ptr)
      ptr->deref();
  }
};

template <typename T, typename PtrTraits = RawPtrTraits<T>, typename RefDerefTraits = DefaultRefDerefTraits<T>> struct Ref {
  typename PtrTraits::StorageType t;

  enum AdoptTag { Adopt };

  Ref() : t{} {};
  Ref(T &t, AdoptTag) : t(&t) { }
  Ref(T &t) : t(&RefDerefTraits::ref(t)) { }
  Ref(const Ref& o) : t(RefDerefTraits::refIfNotNull(PtrTraits::unwrap(o.t))) { }
  Ref(Ref&& o) : t(o.leakRef()) { }
  ~Ref() { RefDerefTraits::derefIfNotNull(PtrTraits::exchange(t, nullptr)); }
  Ref& operator=(T &t) {
    Ref o(t);
    swap(o);
    return *this;
  }
  Ref& operator=(Ref &&o) {
    Ref m(o);
    swap(m);
    return *this;
  }
  void swap(Ref& o) {
    typename PtrTraits::StorageType tmp = t;
    t = o.t;
    o.t = tmp;
  }
  T &get() const { return *PtrTraits::unwrap(t); }
  T *ptr() const { return PtrTraits::unwrap(t); }
  T *operator->() const { return PtrTraits::unwrap(t); }
  operator T &() const { return *PtrTraits::unwrap(t); }
  T* leakRef() { return PtrTraits::exchange(t, nullptr); }
};

template <typename T> Ref<T> adoptRef(T& t) {
  using Ref = Ref<T>;
  return Ref(t, Ref::Adopt);
}

template<typename T> class RefPtr;
template<typename T> RefPtr<T> adoptRef(T*);

template <typename T> struct RefPtr {
  T *t;

  RefPtr() : t(nullptr) { }

  RefPtr(T *t)
    : t(t) {
    if (t)
      t->ref();
  }
  RefPtr(Ref<T>&& o)
    : t(o.leakRef())
  { }
  RefPtr(RefPtr&& o)
    : t(o.t)
  {
    o.t = nullptr;
  }
  RefPtr(const RefPtr& o)
    : t(o.t)
  {
    if (t)
      t->ref();
  }
  RefPtr operator=(const RefPtr& o)
  {
    if (t)
      t->deref();
    t = o.t;
    if (t)
      t->ref();
    return *this;
  }
  ~RefPtr() {
    if (t)
      t->deref();
  }
  Ref<T> releaseNonNull() {
    Ref<T> tmp(*t);
    if (t)
      t->deref();
    t = nullptr;
    return tmp;
  }
  void swap(RefPtr& o) {
    T* tmp = t;
    t = o.t;
    o.t = tmp;
  }
  T *get() const { return t; }
  T *operator->() const { return t; }
  T &operator*() const { return *t; }
  RefPtr &operator=(T *t) {
    RefPtr o(t);
    swap(o);
    return *this;
  }
  operator bool() const { return t; }

private:
  friend RefPtr adoptRef<T>(T*);

  // call_with_adopt_ref in call-args.cpp requires this method to be private.
  enum AdoptTag { Adopt };
  RefPtr(T *t, AdoptTag) : t(t) { }
};

template <typename T> RefPtr<T> adoptRef(T* t) {
  return RefPtr<T>(t, RefPtr<T>::Adopt);
}

template <typename T> bool operator==(const RefPtr<T> &, const RefPtr<T> &) {
  return false;
}

template <typename T> bool operator==(const RefPtr<T> &, T *) { return false; }

template <typename T> bool operator==(const RefPtr<T> &, T &) { return false; }

template <typename T> bool operator!=(const RefPtr<T> &, const RefPtr<T> &) {
  return false;
}

template <typename T> bool operator!=(const RefPtr<T> &, T *) { return false; }

template <typename T> bool operator!=(const RefPtr<T> &, T &) { return false; }

struct RefCountable {
  static Ref<RefCountable> create();
  static std::unique_ptr<RefCountable> makeUnique();
  void ref() {}
  void deref() {}
  void method();
  void constMethod() const;
  int trivial() { return 123; }
  RefCountable* next();
};

template <typename T> T *downcast(T *t) { return t; }

template <typename T> struct CheckedRef {
private:
  T *t;

public:
  CheckedRef() : t{} {};
  CheckedRef(T &t) : t(&t) { t.incrementCheckedPtrCount(); }
  CheckedRef(const CheckedRef &o) : t(o.t) { if (t) t->incrementCheckedPtrCount(); }
  ~CheckedRef() { if (t) t->decrementCheckedPtrCount(); }
  T &get() const { return *t; }
  T *ptr() const { return t; }
  T *operator->() const { return t; }
  operator const T &() const { return *t; }
  operator T &() { return *t; }
};

template <typename T> struct CheckedPtr {
private:
  T *t;

public:
  CheckedPtr() : t(nullptr) {}
  CheckedPtr(T *t)
    : t(t) {
    if (t)
      t->incrementCheckedPtrCount();
  }
  CheckedPtr(Ref<T> &&o)
    : t(o.leakRef())
  { }
  ~CheckedPtr() {
    if (t)
      t->decrementCheckedPtrCount();
  }
  T *get() const { return t; }
  T *operator->() const { return t; }
  T &operator*() const { return *t; }
  CheckedPtr &operator=(T *);
  operator bool() const { return t; }
};

class CheckedObj {
public:
  void incrementCheckedPtrCount();
  void decrementCheckedPtrCount();
  void method();
  int trivial() { return 123; }
  CheckedObj* next();
};

class RefCountableAndCheckable {
public:
  void incrementCheckedPtrCount() const;
  void decrementCheckedPtrCount() const;
  void ref() const;
  void deref() const;
  void method();
  int trivial() { return 0; }
};

template <typename T>
class UniqueRef {
private:
  T *t;

public:
  UniqueRef(T &t) : t(&t) { }
  ~UniqueRef() {
    if (t)
      delete t;
  }
  template <typename U> UniqueRef(UniqueRef<U>&& u)
    : t(u.t)
  {
    u.t = nullptr;
  }
  T &get() const { return *t; }
  operator T&() const { return *t; }
  T *operator->() const { return t; }
  UniqueRef &operator=(T &) { return *this; }
};

class WeakPtrImpl {
private:
  void* ptr { nullptr };
  mutable unsigned m_refCount { 0 };

  template <typename U> friend class CanMakeWeakPtr;
  template <typename U> friend class WeakPtr;

public:
  template <typename T>
  static Ref<WeakPtrImpl> create(T& t)
  {
    return adoptRef(*new WeakPtrImpl(t));
  }

  void ref() const { m_refCount++; }
  void deref() const {
    m_refCount--;
    if (!m_refCount)
      delete const_cast<WeakPtrImpl*>(this);
  }

  template <typename T>
  T* get() { return static_cast<T*>(ptr); }
  operator bool() const { return !!ptr; }
  void clear() { ptr = nullptr; }

private:
  template <typename T>
  WeakPtrImpl(T* t)
    : ptr(static_cast<void*>(t))
  { }
};

template <typename T>
class CanMakeWeakPtr {
private:
  RefPtr<WeakPtrImpl> impl;

  template <typename U> friend class CanMakeWeakPtr;
  template <typename U> friend class WeakPtr;

  Ref<WeakPtrImpl> createWeakPtrImpl() {
    if (!impl)
      impl = WeakPtrImpl::create(static_cast<T>(*this));
    return *impl;
  }

public:
  ~CanMakeWeakPtr() {
    if (!impl)
      return;
    impl->clear();
    impl = nullptr;
  }
};

template <typename T>
class WeakPtr {
private:
  RefPtr<WeakPtrImpl> impl;

public:
  WeakPtr(T& t) {
    *this = t;
  }
  WeakPtr(T* t) {
    *this = t;
  }

  template <typename U>
  WeakPtr<T> operator=(U& obj) {
    impl = obj.createWeakPtrImpl();
  }

  template <typename U>
  WeakPtr<T> operator=(U* obj) {
    impl = obj ? obj->createWeakPtrImpl() : nullptr;
  }

  T* get() {
    return impl ? impl->get<T>() : nullptr;
  }

};

#endif
